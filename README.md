# Notebooks python pandas - Organizacion de datos [2C-2017]

## Requerimientos

* Ubuntu 14.04

## Instalación 
Instalación pip

        $ [sudo] apt-get -y install python-pip

Instalación dependencias pyton

        $ [sudo] apt-get install python-dev
        
Instalar virtualenv para virtualizar la instalación de python

        $ [sudo] pip install virtualenv
        
En adelante llamaremos `ENV` a la carpeta donde virtualizaremos nuestra instalación de python, por lo que ejecutaremos

        $ virtualenv ENV
        
Luego, para la shell que estamos utilizando, activamos la virtualización

        $ cd ENV
        $ source bin/activate
        
Y para desactivarla

        $ deactivate
        
Luego instalar algunas dependencias y liberías de plot

        $ pip install jupyter pandas numpy
        $ pip install matplotlib seaborn bokeh

## Ejecución de Jupyter

Para ejecutar el server Jupyter

    $ jupyter notebook --notebook-dir=/path/to/notebooks/files --port=PORT
    
Se ejecutará un servidor Jupyter en `http://localhost:PORT/` y los notebooks se guardarán en `/path/to/notebooks/files`.

## Pruebas

Ver `example.ipynb`.