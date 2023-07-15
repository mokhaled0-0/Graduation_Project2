from django.shortcuts import render
from django.http import HttpResponse
from face.forms import FaceRecognitionform
from face.machinelearning import faces_detection, face_emotion
from django.conf import settings
from face.models import FaceReconition
import os
import matplotlib.pyplot as plt
from io import StringIO


def index(request):
    form = FaceRecognitionform()
    if request.method == 'POST':
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)

            primary_key = save.pk
            imgobj = FaceReconition.objects.get(pk=primary_key)
            fileroot = str(imgobj.image)

            filepath = os.path.join(settings.MEDIA_ROOT, fileroot)
            results = face_emotion(filepath)
            emoji = []
            for i in results[2]:
                if i >= 0.5:
                    emoji.append("sad")
                else:
                    emoji.append("happy")

            res = zip(results[0], results[1], emoji)
         #   return HttpResponse(res)
            return render(request, 'face_res.html', {'form': form, 'upload': True, 'results': res})

    return render(request, 'index.html', {'form': form, 'upload': False})


def res(request):
    return render(request, 'home/face_res.html')
