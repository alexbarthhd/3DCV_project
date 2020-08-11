# 3DCV_project

self driving toy car

## orga

* next meeting: **Tuesday, 11.08.2020, 11.00**
* **deadline: 02.10.2020**
* report
  * introduction, related work, method, experiments, conclusion
  * template: https://www.ieee.org/conferences/publishing/templates.html
  * min. 8 pages
  * explain taken steps
  * experiments and results
* supervisor: raphael.baumgartner@stud.uni-heidelberg.de

## Tasks

1. Get the hardware from our office and have a look at the tracks in the lab.
2. Setup the hardware and software using the documentation. If you have any problems, feel free to write me an email.
3. Record some training data on one of the tracks.
4. Train one or more neural networks of your choice.
5. Test the model on the other track.
6. Evaluate the model(s) &rarr; Find a good metric (e.g. time per lapse)

### Bonus

* Use Reinforcement Learning (e.g. Dagger)
  * https://arxiv.org/pdf/1011.0686.pdf
  * https://youtu.be/V00npNnWzSU
* Obstacle Detection and Avoidance
  * Train network to detect and drive around obstacles

## Links

* Website: https://www.donkeycar.com/
* Documentation: http://docs.donkeycar.com/
* Hardware: https://www.robocarstore.com/products/donkey-car-starter-kit

## Notes

* kein RGB notwendig
* Strecke bisher: weiß / nicht-farbig, Rand schwarz markiert, Maße etwa 4 x 4 &rarr; Ränder erkennen und Räder entsprechend einstellen, Begrenzung erkennen
* ROS möglich
* Containerisieren ?
* mehr Richtung Deep Learning statt Visualisierung
 - Verschiedene Netzwerke mit einander vergleichen
* OpenCV Ansatz vs Image-processing Netz &rarr; Analyse, Zeit pro Runde, Genauigkeit 
* Wettbewerb gegen andere Gruppen? &rarr; Raphael klärt Interesse, gegebenenfalls Mathematikon Innenhof oder nacheinander in Gruppenlabor
* an sich auch über Laptop statt Raspberry Pi möglich
* Reinforcement Learning auch mit PyTorch möglich
* Raphael klingt begeistert von Reinforcement mit Dagger &rarr; schneller, weniger Daten
* eigenes Betriebssystem? &rarr; Doku, bei Problemen schickt er uns Projektbericht von letztem Jahr
