#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 00:30:10 2017

@author: Julius
"""

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


sentence1 = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])
sentence2 = LabeledSentence(words=[u'another', u'words', u'there'], tags=[u'SENT_2'])
sentence3 = LabeledSentence(words=[u'put', u'something', u'here'], tags=[u'SENT_3'])
sentence4 = LabeledSentence(words=[u'the', u'wheel', u'on', u'the', u'bus', u'go', u'round', u'and', u'round'], tags=[u'SENT_4'])
sentence5 = LabeledSentence(words=[u'round', u'and', u'round'], tags=[u'SENT_5'])
sentence6 = LabeledSentence(words=[u'round', u'and', u'round'], tags=[u'SENT_6'])
sentence7 = LabeledSentence(words=[u'fire', u'in', u'the', u'hole'], tags=[u'SENT_7'])
sentence8 = LabeledSentence(words=[u'roger', u'that'], tags=[u'SENT_8'])
sentence9 = LabeledSentence(words=[u'there', u'is', u'no', u'spoon'], tags=[u'SENT_9'])
sentence10 = LabeledSentence(words=[u'how', u'do', u'you', u'turn', u'this', u'on'], tags=[u'SENT_10'])
sentence11 = LabeledSentence(words=[u'round', u'and', u'round'], tags=[u'SENT_11'])
sentence12 = LabeledSentence(words=[u'diaper', u'and', u'beer'], tags=[u'SENT_12'])
sentence13 = LabeledSentence(words=[u'apple', u'and', u'orange'], tags=[u'SENT_13'])

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, 
             sentence7, sentence8, sentence9, sentence10, sentence11, sentence12, sentence13]


model = Doc2Vec(alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
for epoch in range(200):
    ## Range - 20, poor performance; 200, seems great
    if epoch % 20 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(sentences, total_examples=len(sentences), epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    
model.save('doc2vecPractice.model')
print(model.docvecs.most_similar(["SENT_5"]))
print(model.docvecs.most_similar(["SENT_2"]))