from Dashboard import cargar_datos
from Dashboard import crear_graficos
from Dashboard import crear_aplicacion
import os

data = cargar_datos()
graficos = crear_graficos(data)
app = crear_aplicacion(graficos)
app = app.server

if __name__ == '__main__':
    app.run(debug=False)