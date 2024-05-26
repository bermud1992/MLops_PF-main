# MLops_P2

0. Posterior a activar la VPN de la univeridad, conectese a la maquina virtual con las credenciales del grupo 2.
1. ejecutar el comando:  
	```url
	docker compose up
2. ingresar a la url:
    ```url
    http://10.43.101.151:8086/login
	```
3. ingresar las siguientes credenciales en la ventana de inicio de sesion <br />
	Usuario: admin <br />
	Password: supersecret <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_0.png)
4. cree un nuevo bucket llamado mlflow
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_1.png)
    ![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_2.png)
    ![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/minio_3.png)
5. ingresar a la url:
    ```url
    http://10.43.101.151:8080/
	```
	Ingresar las siguientes credenciales en la ventana de inicio de sesion <br />
	Usuario: airflow <br />
	Password: airflow <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/airflow_0.png) <br />
6. activar el DAG "Data_fetch_dag" el cual traera y guardara un batch de data cada 5 mins
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/airflow_1.png)
5. activar el DAG "Model_train_dag" el cual correra el modelo de ML, este se volvera a correr automaticamente cada 50 mins cuando se tenga toda la data  <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/airflow_2.png) <br />
	si desea realizar una corrida adicional con la data disponible hasta el momento sin esperar a tener toda la data, ingrese al dag y corralo a discreción: <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/airflow_3.png) <br />
6. ingresar a la url:
    ```url
    http://10.43.101.151:8087/
	```
	Hacer click en "mlflow_tracking_examples" a la izquierda y dar click a la corrida mas actual <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_0.png) <br />
	Registre el modelo "modelo_base" <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_1.png) <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_2.png) <br />
7. Haga upgrade de la ultima version del modelo a produccion
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_3.png) <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_4.png) <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_5.png) <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/mlflow_6.png) <br />
8. Acceder a la url:   
    ```url
    http://10.43.101.151:8088/docs
	```
	Realizar una prediccion a traves del metodo POST "predict" con el nombre del modelo "modelo_base" <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/api_0.png) <br />
	![alt text](https://github.com/marinho14/MLops_P2/blob/main/images/api_1.png) <br />
