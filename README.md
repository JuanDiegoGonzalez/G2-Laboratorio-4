# Laboratorio 4

## Integrantes:
* Juan Felipe Peña Criado (201426463) 

* Juan Diego González Gómez (201911031) 

* Sergio Ramírez Vélez (201714577) 

## Instrucciones de instalación y despliegue del API

1. Clonar o descargar el repositorio:

  ```shell
  git clone https://github.com/JuanDiegoGonzalez/G2-Laboratorio-4.git
  ```

2. Abrir el proyecto en Android Studio o el IDE de su preferencia

3. Abrir una terminal e instalar los requerimientos:

  ```shell
  pip install -r requirements.txt
  ```

4. Ejecutar el programa en la terminal:

  ```shell
  uvicorn main:app --reload
  ```

5. Se podrá verificar el funcionamiento del programa ingresando desde un navegador a la URL http://127.0.0.1:8000/ (se debe visualizar un objeto JSON: "{"Hello":"World"}")

## Instrucciones del funcionamiento del API:

Opción A - Realizar peticiones desde el navegador:

1. Con el programa ejecutándose (ver sección anterior), ingresar desde un navegador a la URL: http://127.0.0.1:8000/docs#/

2. Hacer click en las secciones POST (rutas /predict y /r2)

3. Hacer click en el botón "Try it out"

4. Modificar el objeto JSON (ver ejemplos válidos en el documento enviado por la actividad de Bloque Neon)

5. Hacer click en el botón "Execute"

6. Se obtendrá la respuesta en la sub-sección "Response body" de la sección "Server response"

Opción B - Hacer peticiones desde Postman:

1. Abrir Postman

2. Crear una nueva petición

3. Seleccionar el método POST

4. Ingresar la URL http://127.0.0.1:8000/predict o http://127.0.0.1:8000/r2

5. En la sección Body, seleccionar "raw". Luego de esto, aparecerá una nueva opción con un valor por defecto "Text"

6. En la nueva opción que apareció, seleccionar "JSON"

7. Ingresar el objeto JSON (ver ejemplos válidos en el documento enviado por la actividad de Bloque Neon)

8. Hacer click en el botón "Send"

9. Se obtendrá la respuesta en la sub-sección "Body" de la sección "Response"
