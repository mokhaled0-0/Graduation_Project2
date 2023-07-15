from django.shortcuts import render, redirect
from django.http import HttpResponse
from .sentiment_analysis_code import sentiment_analysis_code
from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly.express as px
import pandas as pd
from textblob import TextBlob
from .helpers import *


def home(request):
    return render(request, 'home/excel_input.html')


def analyze(request):
    if request.method == "POST":
        analyse = sentiment_analysis_code()
        excel_file = request.FILES.get("ex_file")
        if excel_file == None:
            return HttpResponse("there is no file has been uploaded")
        custom = readex(request)
        if custom == None:
            return HttpResponse("file is empty or unsupported format")
        list_of_tweets_and_sentiments = []
        sent = {}
        sentiment = []
        for i in custom:
            s = analyse.predicts([i])
            if abs(s[3] - .61) <= .01:
                s = (s[0], s[1], .5, .5)
            st = ""
            if s[3] >= .55:
                st = 'positive'
            elif s[3] >= .45:
                st = 'neutral'
            else:
                st = "negative"
            list_of_tweets_and_sentiments.append((i, st))
            sent[st] = sent.get(st, 0) + 1
            sentiment.append(st)
            colorss = sent.keys()
        fig = px.pie(names=colorss, values=sent.values(), color=colorss,
                     color_discrete_map={'positive': '#39e75f',
                                         'negative': '#d91515',
                                         'neutral': '#89bdee'})
        fig.update_layout(
            {'paper_bgcolor': 'rgba(0, 0, 0 ,0)'},
        )
        fig.update_layout(width=int(620))
        p = fig.to_html()
        custom = zip(custom, sentiment)
        arg = {'p': p, 'texts': custom}
        return render(request, 'home/res2.html', arg)
    else:
        return HttpResponse("error")
