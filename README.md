# O projektu

## Cilj projekta
Cilj projekta predstavlja obezbeđivanje frameworka za lako 
pravljenje object detection modela optimizovanog za korišćenje
na velikom broju slabih uređaja i exportovanom u tflite 
formatu. Ovo bi omogućilo korisniku da vrlo brzo kreira 
sopstveni object detection model koji bi uz par jednostavnih 
izmena i učitavanjem odgovarajućeg skupa podataka mogao da 
prepoznaje klase koje sam korisnik izabere. 

Sporedni ciljevi uključuju:

- pravljenje skripti za jednostavno kreiranje seta podataka 
kompatibilnog sa treningom modela, 
- pripremanje modela za prepoznavanje položaja ruke Like, 
Palm, Serbia
- pravljenje skripte za prepoznavanje datih klasa na video 
stream-u ucitanog sa kamere Raspberry Pi uređaja


## Uputstvo za korišćenje
U zavisnosti od sitema na kome se pokreću skripte, neophodno 
je pokrenuti jednu od dve skripte za instaliranje "requirements"

Za Linux:

    sh ./linux_setup_venv.sh
    sh ./linux_requirements.sh

Za Windows:

    .\windows_setup_venv.bat
    .\windows_requirements.bat

### Korak po korak kroz projekat

_Prvo treba izgenerisati dataset za treniranje modela 
pokretanjem "image_collection.py" skripte_

_Sledeće što je neophodno je da se pokrene programa "labelImg" iz konzole 
(biblioteka koja je instalirana pokretanjem requirements fajla)_

_Nakon sto smo pripremili dataset, potrebno je dobaviti 
["model garden"](https://github.com/tensorflow/models) da bi 
imali neophodne skripte za rad sa object detection APIs 
(sadrzaj kloniranog repozitorijuma kopirati u direktorijum 
/tensorflow/models)_

_Sledeći korak je kloniranje 
[Protobuf repozitorijuma](https://github.com/protocolbuffers/protobuf/releases)
i kopiranje njegovog sadrzaja unutar /tensorflow/protoc direktorijuma_

    $ ./tensorflow/protoc/bin/protoc.exe ./tensorflow/models/research/object_detection/protos/*.proto --python_out=.

_Zatim treba da instaliramo "COCO API", u slučaju da nije pravilno
povučen kao dependency kroz instalaciju prethodnih biblioteka, i
to pokretanjem sledećih komandi:_

    $ pip install cython
    $ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

_Konačno, treba da osposobimo "Object Detection API":_

    $ cp ./tensorflow/models/research/object_detection/packages/tf2/setup.py ./tensorflow/models/research/
    $ python -m pip install --use-feature=2020-resolver .

_Na kraju, možemo proveriti validnost instalacije pokretanjem sledeće komande koja bi trebala
u poslednjoj liniji da istampa OK na konzolu:_

    $ python ./tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py

### _Treniranje_

_Da bi se počelo sa treniranjem potrebno je sa 
[repozitorijuma TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) 
preuzeti odgovarajući istreniran model na kome će se 
raditi prilagođavanje. Za potrebe našeg istraživanja 
koristili SSD MobileNet v2 320x320 
(preuzeti ZIP treba raspakovati na putanju 
/tensorflow/workspace/pre-trained-models)_

_Treniranje na sopstvenom datasetu će se pokrenuti 
izvršavanjem skripte /tensorflow/workspace/training.py_

_Potrebno je na kraju napomenuti da je neophodno 
izmeniti niz labels unutar constants.py tako da
odgovara labelama koje se nalaze na generisanim slikama_
