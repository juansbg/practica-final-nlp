# Merger ML

Predicción de fusiones y adquisiciones bancarias basadas en NLP sobre los annual report.

## Deployment

NO clonar todo el Git (son muchos gigas por el dataset de entrenamiento). Lo único que hay que descargar es la carpeta 'Dash', instalar los requirements.txt de la carpeta:
```pip
  pip install -r requirements.txt
```
Y luego ejecutar el script llamado 'dash_bank.py':
```pip
  python dash_bank.py
```
Y probarlo. OJO, por solucionar problemas de decodificar bytes subidos, lo que hace el dash es buscar el PDF subido en la carpeta 'PDFs', asi que solo funcionará si el PDF de encuentra en esa carpeta y si su nombre tiene el siguiente estilo:
```pip
  BANCO_AÑO.pdf
```
