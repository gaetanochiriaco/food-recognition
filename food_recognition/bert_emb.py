import torch
import pandas as pd
from transformers import BertTokenizer, BertModel

def get_emb(food_label,tokenizer,model):

  tokenized_text = tokenizer.tokenize(food_label)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1] * len(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  
  with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
  
  return hidden_states[-1].mean(dim=1)

def get_bert_embeddings(label_path, model_name = "bert-large-uncased"):

    labels = pd.read_csv(label_path,encoding="latin1",sep=";")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name,
                                  output_hidden_states = True)
    
    bert_model.eval()


    for i in range(labels.shape[0]):
      if i == 0:
        bert_emb = get_emb(labels.Category[i],tokenizer,bert_model)

      else:
        bert_emb = torch.cat([bert_emb,get_emb(labels.Category[i],tokenizer,bert_model)], dim = 0)

    return bert_emb