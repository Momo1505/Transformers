import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split

from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset,causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm
import warnings

def greedy_decode(model,source,source_mask,src_tokenizer,tgt_tokenizer,max_len,device):
    sos_token = tgt_tokenizer.token_to_id("[SOS]")
    eos_token = tgt_tokenizer.token_to_id("[EOS]")

    # compute once the encoder output to be reused later for inference
    encoder_output = model.encode(source,source_mask)

    # initialize the decoder input with SOS
    decoder_input = torch.empty((1,1)).fill_(sos_token).type_as(source).to(device)
    while True:
        if decoder_input.size(1)==max_len:
            break
        # build mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        # compute the model output
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

        # get the next token
        proj = model.project(out[:,-1])

        # select the token with max probability
        _,next_word = torch.max(proj,dim=1)

        decoder_input = torch.cat([
            decoder_input,
            next_word.unsqueeze(0)
        ], dim=1)

        if next_word == eos_token:
            break

    return decoder_input.squeeze(0)

def run_validation(model,val_ds,src_tokenizer,tgt_tokenizer,max_len,device,print_msg, gloabal_state,writer, num_examples=2):
    model.eval()
    count = 0

    console_width = 80

    with torch.no_grad():
        for batch in val_ds:
            count+=1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1 , "Batch size must be 1 for validation"

            model_out = greedy_decode(model,encoder_input,encoder_mask,src_tokenizer,tgt_tokenizer,max_len,device)

            source_text = batch["src_text"]
            target_text = batch["tgt_text"]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            # print to console
            print_msg("-"*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break


def get_all_sentences(ds,lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # create the tokenizer and tell which model to use
        tokenizer.pre_tokenizer = Whitespace() # how to split the sentences
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2) # instantiate the trainer
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("opus_books",f"{config['lang_src']}-{config['lang_tgt']}",split="train")

    #build tokenizers
    src_tokenizer = get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tgt_tokenizer = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,src_tokenizer,tgt_tokenizer,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw,src_tokenizer,tgt_tokenizer,config["lang_src"],config["lang_tgt"],config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = src_tokenizer.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tgt_tokenizer.encode(item["translation"][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f"Max length of source sentence {max_len_src}")
    print(f"Max length of target sentence {max_len_tgt}")

    train_dl = DataLoader(train_ds,config["batch_size"],shuffle=True)
    val_dl = DataLoader(val_ds,batch_size=1, shuffle=True)
    return train_dl,val_dl, src_tokenizer, tgt_tokenizer

def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config["seq_len"],config["seq_len"])
    return model

def train_model(config):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True,exist_ok=True)

    train_dl,val_dl, src_tokenizer, tgt_tokenizer = get_ds(config)

    model = get_model(config, src_tokenizer.get_vocab_size(),tgt_tokenizer.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps=1e-9)

    intial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config,config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        intial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

    for epoch in range(intial_epoch,config["num_epochs"]):

        batch_iterator = tqdm(train_dl,desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch,1,1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch,1,seq_len, seq_len)

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input,encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device) # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size) 
            loss = loss_fn(proj_output.view(-1,tgt_tokenizer.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # compute the gradients
            loss.backward()

            #update the weights

            optimizer.step()
            optimizer.zero_grad()

            global_step +=1
        
        run_validation(model,val_dl,src_tokenizer,tgt_tokenizer,config["seq_len"],device, lambda msg: batch_iterator.write(msg), global_step,writer)


        # save the model
        model_filename = get_weights_file_path(config,f"{epoch:0.2d}")

        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        },model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
