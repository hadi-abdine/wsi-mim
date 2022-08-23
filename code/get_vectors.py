from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import AutoModelForSeq2SeqLM, RobertaModel, AutoConfig, BertModel, DebertaTokenizer, DebertaModel, DebertaConfig
import torch
import numpy as np
import random
import nltk
import argparse
from os import listdir
from os.path import isfile, join
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os


parse = argparse.ArgumentParser()
parse.add_argument('--path', type=str, default='contexts/txt-train/')
parse.add_argument('--split', type=str, default='train')
parse.add_argument('--model', type=str, default='bart')
parse.add_argument('--save', type=str, default='contexts/vectors/')
parse.add_argument('--transform', type=str, default='')
parse.add_argument('--layer', type=int, default=7)
parse.add_argument('--cls', type=int, default=0)
args = parse.parse_args()


def replace_with_lemma(lemma, pos, sentence, lemmatize_target=False):
    s = ''
    for token in sentence.replace('-',' ').replace('=', ' =').replace('.', ' ').replace("'","' ").lower().split():
        if lemma==wnl.lemmatize('color', pos) and token=='colour':
            if lemmatize_target: s += lemma + ' '
            else: s += token + ' '
        elif lemma==wnl.lemmatize('lie', pos) and (wnl.lemmatize(token, pos)=='lay' or wnl.lemmatize(token, pos)=='lain' or token=='lah'):
            if lemmatize_target: s += lemma + ' '
            else: s += token + ' '
        elif lemma==wnl.lemmatize('figure', pos) and (token=='figger' or token=='figgered'):
            if lemmatize_target: s += lemma + ' '
            else: s += token + ' '
        elif wnl.lemmatize(token, pos) == lemma:
            if lemmatize_target: s += lemma + ' '
            else: s += token + ' '
        else:
            s += token + ' '
    return s


print('Creating ', join(args.save, args.split+"_vectors_"+args.model+str(args.layer)+"_"+args.transform+".txt"))
if not os.path.exists(args.save):
    os.makedirs(args.save)


files = [f.split('.')[0]+'.'+f.split('.')[1] for f in listdir(args.path) if isfile(join(args.path, f))]
wnl = WordNetLemmatizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if args.model == 'barthez':
    tokenizer = AutoTokenizer.from_pretrained("moussaKam/mbarthez")
    model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/mbarthez")
    special = "▁"
if args.model == 'deberta':
    config = DebertaConfig.from_pretrained("microsoft/deberta-xlarge-mnli", output_hidden_states=True)
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
    model = DebertaModel.from_pretrained("microsoft/deberta-xlarge-mnli", config=config)
    special = "Ġ"
if args.model == 'roberta':
    config = AutoConfig.from_pretrained("roberta-large", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained("roberta-large", config=config)
    special = "Ġ"
if args.model == 'bert':
    config = AutoConfig.from_pretrained("bert-large-uncased", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    model = BertModel.from_pretrained("bert-large-uncased", config=config)
    special = "##"
if args.model == 'bart':
    config = AutoConfig.from_pretrained("facebook/bart-large-mnli", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModel.from_pretrained("facebook/bart-large-mnli", config=config)
    special = "Ġ"

model.to(device)
model.eval()
f1 = open(join(args.save, args.split+"_vectors_"+args.model+str(args.layer)+"_"+args.transform+".txt"), 'w')


for i,file in enumerate(files):
    f = open(args.path+file+'.txt', 'r')
    sentences = f.readlines()
    f.close()
    mars_vectors = []
    skip_list = []
    wordp = file.split('.')[0]
    pos = file.split('.')[1]
    if pos=='j':
        pos = 'a'
    word = wnl.lemmatize(wordp, pos)
    print(wordp, ': ',i+1, '/', len(files))
    count_skip = 0
    pieces = 0

    for sentence in sentences:
        
        sentence = replace_with_lemma(word, pos, sentence)
        if args.model == 'barthez':
            sentence = ' '.join(nltk.word_tokenize(sentence.lower().replace('-',' '), language='french'))
        else:
            sentence = ' '.join(nltk.word_tokenize(sentence.lower().replace('-',' ').replace('=', ' =').replace('.', ' ').replace("'","' ")))
        encoded_input = tokenizer.encode(sentence, return_tensors='pt', add_special_tokens=True, max_length=500, truncation=True).to(device)

        with torch.cuda.amp.autocast():
            hidden_states = model(encoded_input)
        if args.model=='deberta':
            token_embeddings = hidden_states[1][args.layer]
        elif  args.model=='bart':   
            token_embeddings = hidden_states['encoder_hidden_states'][args.layer]
        else:
            token_embeddings = hidden_states[2][args.layer]

        word_instances = []
        tokens = [token_str for token_str in tokenizer.convert_ids_to_tokens(encoded_input[0])]
        if args.cls == 0:
            #  args.model != 'deberta' and
            if args.model != 'bert' and args.model != 'bert2':
                tokens[1] = special+tokens[1]       
            i = 1
            while i < len(tokens):
                if tokens[i]=='</s>' or tokens[i]==special+'</s>' or tokens[i]=='[SEP]' or tokens[i]==special+'[SEP]':
                    break

                elif wordp=='color' and (tokens[i][1:]=='colour' or tokens[i]=='colour'):
                    word_instances.append([i,])
                elif wordp=='lie' and (wnl.lemmatize(tokens[i][1:])=='lay' or wnl.lemmatize(tokens[i][1:])=='lain' or wnl.lemmatize(tokens[i])=='lain' or wnl.lemmatize(tokens[i])=='lay'):
                    word_instances.append([i,])
                elif wordp=='figure' and (tokens[i][1:]=='figger' or tokens[i]=='figger'):
                    word_instances.append([i,])
                else:
                    if args.model != 'bert' and args.model != 'bert2':
                        if word == wnl.lemmatize(tokens[i][1:], pos) and (tokens[i+1][0] == special or tokens[i+1]=='</s>'):
                            word_instances.append([i,])

                        else: 
                            tmp_word = tokens[i]
                            tmp_i = [i]
                            while tokens[i+1][0] != special and tokens[i+1]!='</s>' and tokens[i+1]!='[SEP]' and tokens[i+1]!=special+'</s>':
                                i += 1
                                tmp_word += tokens[i]
                                tmp_i.append(i)
                            if wnl.lemmatize(tmp_word[1:], pos) == word:
                                word_instances.append(tmp_i)
                                pieces += 1
                            if wordp=='lie' and (tmp_word[1:]=='lain' or tmp_word[1:]=='lah'):
                                word_instances.append(tmp_i)
                                pieces += 1
                            if wordp=='figure' and (tmp_word[1:]=='figger' or tmp_word[1:]=='figgered'):
                                word_instances.append(tmp_i)
                                pieces += 1
                    else:
                        if word == wnl.lemmatize(tokens[i], pos) and (tokens[i+1][0:2] != special or tokens[i+1]=='[SEP]' or tokens[i+1]=='##[SEP]'):
                            word_instances.append([i,])
                        else:
                            tmp_word = tokens[i]
                            tmp_i = [i]
                            if len(tokens[i+1]) < 2:
                                i += 1
                                continue
                            while tokens[i+1][0:2] == special and tokens[i+1] != '[SEP]' and tokens[i+1] != '##[SEP]':
                                i += 1
                                tmp_word += tokens[i][2:]
                                tmp_i.append(i)
                            if wnl.lemmatize(tmp_word, pos) == word:
                                word_instances.append(tmp_i)
                                pieces += 1
                            if wordp=='lie' and (tmp_word=='lain' or tmp_word=='lah'):
                                word_instances.append(tmp_i)
                                pieces += 1
                            if wordp=='figure' and (tmp_word=='figger' or tmp_word=='figgered'):
                                word_instances.append(tmp_i)
                                pieces += 1
                i += 1   
            if len(word_instances)==0:
                count_skip += 1
                # print(tokens)
                mars_vectors.append(['skip'])
                skip_list.append(['skip'])
                continue

            index = random.choice(word_instances)
        else:
            index = [0]
        
        vectors = [token_embeddings[-1][i].detach().cpu().numpy() for i in index]
        vector = np.mean(vectors, axis=0)

        mars_vectors.append(vector)

        skip_list.append(['good'])
    print(wordp,': ',count_skip,' skipped', '  and   ', pieces, ' joined pieces')
    
    for line in mars_vectors:
        f1.write(wordp+'.'+file.split('.')[1]+" - "+','.join([str(w) for w in line])+'\n')

        
f1.close()

