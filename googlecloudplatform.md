## Google Cloud Platform

Google Cloud Hosting, and or Compute Engine, is just one of many services provided by Google Cloud Platform. They also offer their App Engine, storage, DNS, CDN, and a multitude of other services and products. Google Cloud Platform consists of a set of physical servers, as well as virtual resources like virtual machines (VMs) that are contained in Google’s data centers around the globe. We use Google Cloud Platform’s multi-regional deployment mode. Cloud computing is the delivery of computing services for servers, storage, databases, networking, software, analytics, artificial intelligence and moreover deployed in the Internet (“the cloud”) to offer faster innovation and flexible resources.

![GCP Benefits:](/images/Images/GCP_benifits.png)<br/>
&nbsp;&nbsp;
![ML Architecture:](/images/Images/ml_architecture.png)<br/>

```javascript
Cloud SQL Instance for Database(SQL Standard):
M/C type : db-n1-standard-4
RAM(GB) : 15
Max Storage Capacity: 50 GB
Max Connections: 4000

We will require Cloud type as : My SQL and Second Generation.

Cloud Storage as a Common database as of now are using our google drive to store and retrieve the data.

Compute Engine Instance for the Capsule Network project:
M/C type : n1-standard-8
Virtual CPU: 8
Memory: 50 GB
```

![GCP Benefits:](/images/Images/GCP.png)<br/>
&nbsp;&nbsp;
![ML Architecture:](/images/Images/Bucket_Cloud.png)<br/>
&nbsp;&nbsp;
![ML Architecture:](/images/Images/Database_Cloud_SQL.png)<br/>

We have implemented the Cloud SQL Database in GCP and have to import the SQL database for the database students created by the AI Skunkworks. And implementing the Cloud ML API and Data Flow API are imported into the GCP to help perform Machine Learning Algorithms.
Also create a instance for Jupyter notebook in the SSH shell of the instance to install all the dependcies like Python, TensorFlow, Keras, numpy, pandas and sckitlean to run the project.
