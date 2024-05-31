[![Texto alternativo](https://img.youtube.com/vi/wmhRgMyUyBs/0.jpg)](https://youtu.be/wmhRgMyUyBs)

# MLOps Proyecto Final

1. Conectese a la VPN de la Universidad mediante arpuj Javeriana
   
2. Por favor ubicarse sobre el directorio del proyecto 3 con el comando
   ```url
	cd /home/estudiante/MLOps_P3
3. Una vez alli ejecutar el comando:  
	```url
	docker compose up
4. Ingresar a la url:
    ```url
    http://10.43.101.151:8086/login
	```
5. Ingresar las siguientes credenciales en la ventana de inicio de sesion <br />
	Usuario: admin <br />
	Password: supersecret <br />
 
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_0.png)

6. En caso de no existir, cree un nuevo bucket llamado mlflow
   
    ![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_1.png)

    ![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_2.png)
   
    ![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_3.png)
   
7. Ingresar a la url:
    ```url
    http://10.43.101.151:8080/
	```
	Ingresar las siguientes credenciales en la ventana de inicio de sesion <br />
	Usuario: airflow <br />
	Password: airflow <br />
 
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/airflow_0.png) <br />

8. Ingrese a la interfaz grafica de Airflow en la direccion: http://10.43.101.151:8080/home y haga clic en el boton que dispara el DAG, si lo desea puede seguir el desarrollo de la ejecucion haciendo clic en la celda last run que ahora tiene la fecha actual del ultimo run enviado.

   ![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/airflow1.png)  <br />
   
   ![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/airflow2.png)  

Una vez esta ejecucion termine con las 3 cajas del grafo en status success el modelo estara disponible en mlflow.

9. Ingrese a la interfaz de mlflow a traves de la direccion http://10.43.101.151:8087/#/models en esta ventanaencontrara listados los modelos generados y vera que ya cuentan con el alias de produccion que permite distinguirlos de los otros modelos.

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflow1.png) 

 En caso de no tener modelos registrados, el procedimiento es el siguiente:
 
![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr1.png)  <br />

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr2.png)  <br />

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr3.png)  <br />

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr4.png)  <br />

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr5.png)  <br />

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr6.png)  <br />

![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr7.png)  <br />


![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/mlflowr8.png)  <br />

10. Ingresar a la url http://10.43.101.151:8082/ , en esta direccion se encuentra alojada la aplicacion streamlit en la cual se encuentra la siguiente interfaz grafica:

   ![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/streamlit2.png) 

   En esta interfaz puede modificar los datos de prediccion o dejar los ya existentes, una vez ha revisado / modificado los datos para predecir puede hacer clic en el boton "realizar prediccion". El sistema devolvera una estructura Json donde encontrara el nombre del modelo utilizado y el valor predicho como se observa en la imagen:

   ![alt text](https://github.com/bermud1992/MLOps_P3/blob/main/images/streamlit1.png) 
