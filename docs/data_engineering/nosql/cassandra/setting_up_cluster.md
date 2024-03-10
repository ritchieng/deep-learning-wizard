---
comments: true
---

# Setting up Cassandra Multi-node Cluster

## Install Apache Cassandra 

!!! warning "Patience is key initially"
	It will be quite tedious to many people in setting up a multi-node Cassandra cluster. You are likely to face a lot of problems. But with a strong community and this tried-and-proven guide, you should be ok! Just be patient.

	Remember when you're asking for help on StackOverflow, here or anywhere else, do paste the logs to make everyone's life easier to help you debug quickly. 

	All you need to do is to run this base command and paste the logs.

	```
	cat /var/log/cassandra/system.log
	```

Here, we will install all of the required Debian packages. To ensure you've the latest version of Cassandra, please use the official link at [cassandra.apache.org/download](http://cassandra.apache.org/).

Bash Commands
```
echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list

curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -

sudo apt-get update

sudo apt-get install cassandra
```

This would get you up and running with Cassandra v3.11.

## Install DataStax Python Cassandra Driver

We need to install the Python Cassandra Driver so we can easily interact with our Cassandra Database without using the CQLSH (this is Cassandra's shell to run CQL commands to perform CRUD operations) commandline directly.

!!! question "What is CRUD?"
	CRUD is the acronym of CREATE, READ, UPDATE, and DELETE. For databases, we typically have these 4 categories of operations so we can create new data points, update, read or delete them in our database.


Bash Commands
```
pip install cassandra-driver
```

This is strictly for installation on Linux platforms, refer to the [official website](https://datastax.github.io/python-driver/installation.html) for more details.

## Single Node Cassandra Cluster

Before we venture into the cool world of multi-node cluster requiring many servers, we will start with a single node cluster that only requires a single server (desktop/laptop).

Once you've installed everything so far, you should run the following.

Bash commands

```
sudo service cassandra start
sudo nodetool status
```

And this will print out something like that:

```
Datacenter: datacenter1
=============================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address       Load       Tokens       Owns (effective)  Host ID                               Rack
UN  127.0.0.1     318.78 KiB  256          100.0%            g5ac4c9-99b7-65d-24cfd82524f9      rack1

```

That's it, we've built our single node Cassandra database.


## Multi-node Cassandra Cluster

Remember it's standard to have at least 3 nodes, and in a basic 3 separate server configuration (3 separate desktops for non-enterprise users). Because of this, we will base this tutorial on setting up a 3-node cluster. But the same steps apply if you want to even do 10000 nodes.

!!! tip "Installing Sublime critical"
	Sublime is a text editor, and it would help if you're not familiar with VIM which I use frequently. Honestly using Sublime to edit all the following files we will edit is much easier, trust me! 

	To install Sublime run the following bash commands in sequence:
	```
	wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -

	echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list

	sudo apt-get update

	sudo apt-get install sublime-text
	```

	Now check if it works by running this command in your bash:
	```
	subl
	```

	This should open Sublime text editor! You're now ready.

Before we move on to the steps to set up each node, we need the IP address of all 3 servers. This is simple, just run the following bash command:

```
ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p'
```

This would return the server's IP address that we need for example:

```
server_1_ip: 112.522.6.61
server_2_ip: 112.522.6.62
server_3_ip: 112.522.6.63
```

We will be using these IPs as a base for configuration in the subsequent sections. Take note yours would differ and you need to change accordingly.

### Steps Per Node

!!! warning "Critical Section You Need To Repeat"
    This is a critcal section. On **EVERY** server, you need to repeat all the steps shown in this section. In our case of a 3 node cluster, using 3 servers, we need to repeat this 3 times.


#### Step 1: Modify Cassandra Configuration Settings


Run the following bash to edit the configuration file for server 1 (112.522.6.61):

```
cd /etc/cassandra
subl cassandra.yaml
```

Now you need to find the following fields and change them accordingly.


```
cluster_name: 'CassandraDBCluster'


seed_provider:
  - class_name: org.apache.cassandra.locator.SimpleSeedProvider
    parameters:
         - seeds: "112.522.6.61, 112.522.6.62, 112.522.6.63"


listen_address: 112.522.6.61


rpc_address: 112.522.6.61


endpoint_snitch: GossipingPropertyFileSnitch

auto_bootstrap: true

```

#### Step 2: Modify Rack Details

Run the following bash command to edit the rack details:

```
cd /etc/cassandra
subl cassandra-rackdc.properties
```

And change the following fields to this:
```
dc=datacenter1
```

#### Step 3: Check Status

Restart service with the following bash commands:

```
sudo rm -rf /var/lib/cassandra/data/system/*
sudo service cassandra restart
sudo nodetool status
```

Here you would see an output like this:

```
Datacenter: datacenter
=============================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address       Load       Tokens       Owns (effective)  Host ID                               Rack
UN  112.522.6.61  318.78 KiB  256          100.0%            f5c84c9-99b7-45d-8856-24cfd82523f9  rack1
UN  112.522.6.62  206.42 KiB  256          100.0%            ac2f24da-1b2c-4e3-8a75-0e28ec366a4  rack1
UN  112.522.6.63  338.34 KiB  256          100.0%            55a227d-ffbe-45d5-8698-60c374b3a6b  rack1

```

#### Step 4: Configure Firewall IP Settings


Run the following bash commands

```
sudo apt-get install iptables-persistent

sudo iptables -A INPUT -p tcp -s 112.522.6.62 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT
sudo iptables -A INPUT -p tcp -s 112.522.6.63 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

sudo bash -c "iptables-save > /etc/iptables/rules.v4"

sudo netfilter-persistent reload

sudo nodetool status
```

### Repeat Steps 1 to 4

You have to repeat all 4 steps for the other 2 servers but **step 1** and **step 4** requires slightly different settings. For the other servers, you need to basically change the IP to the server's IP. And for the IP firewall settings, you need to allow entry from the other 2 servers instead of itself.


####  Server 2 Example for Step 2

```
cluster_name: 'CassandraDBCluster'


seed_provider:
  - class_name: org.apache.cassandra.locator.SimpleSeedProvider
    parameters:
         - seeds: "112.522.6.61, 112.522.6.62, 112.522.6.63"


# ONLY THIS LINE CHANGES
listen_address: 112.522.6.62

# ONLY THIS LINE CHANGES
rpc_address: 112.522.6.62


endpoint_snitch: GossipingPropertyFileSnitch

auto_bootstrap: true

```

####  Server 3 Example for Step 2

```
cluster_name: 'CassandraDBCluster'


seed_provider:
  - class_name: org.apache.cassandra.locator.SimpleSeedProvider
    parameters:
         - seeds: "112.522.6.61, 112.522.6.62, 112.522.6.63"


# ONLY THIS LINE CHANGES
listen_address: 112.522.6.63

# ONLY THIS LINE CHANGES
rpc_address: 112.522.6.63


endpoint_snitch: GossipingPropertyFileSnitch

auto_bootstrap: true

```

####  Server 2 Example for Step 4

```
sudo apt-get install iptables-persistent

# ONLY THESE LINES CHANGE TO ALLOW COMMUNICATION WITH SERVER 1 AND 3
sudo iptables -A INPUT -p tcp -s 112.522.6.61 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT
sudo iptables -A INPUT -p tcp -s 112.522.6.63 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

sudo bash -c "iptables-save > /etc/iptables/rules.v4"

sudo netfilter-persistent reload

sudo nodetool status
```

####  Server 3 Example for Step 4

```
sudo apt-get install iptables-persistent

# ONLY THESE LINES CHANGE TO ALLOW COMMUNICATION WITH SERVER 1 AND 2
sudo iptables -A INPUT -p tcp -s 112.522.6.61 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT
sudo iptables -A INPUT -p tcp -s 112.522.6.62 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

sudo bash -c "iptables-save > /etc/iptables/rules.v4"

sudo netfilter-persistent reload

sudo nodetool status
```

## Dead Node Fix


When your server/desktop restarts, you may face a dead node. You need replace the address with the same IP for it to work.

```
cd /etc/cassandra
subtl cassandra-env.sh
```

Add the following line in the last row assuming server 1 is dead where the IP is 112.522.6.61
```
JVM_OPTS="$JVM_OPTS -Dcassandra.replace_address=112.522.6.61"
```

Then run the following bash commands

```
sudo rm -rf /var/lib/cassandra/data/system/*
sudo service cassandra restart
sudo nodetool status

```

You might have to wait awhile and re-run `sudo nodetool status` for the DB to get up and running.


## Summary

We have successfully set up a 3-node Cassandra cluster DB after all these steps. We will move on to interacting with the cluster with CQLSH and the Python Driver in subsequent guides.

## Citation
If you have found these useful in your research, presentations, school work, projects or workshops, feel free to cite using this DOI.

[![DOI](https://zenodo.org/badge/139945544.svg)](https://zenodo.org/badge/latestdoi/139945544) 