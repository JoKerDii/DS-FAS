# SQL

MySQL is a database management system.

Database is a structured set of computerized data with an accessible interface.

SQL: "Structured query language", the language we use to talk to our databases.

## 1. Creating Databases and Tables

```sql
mysql-ctl cli
show databases;
```

```sql
CREATE DATABASE hello_world_db;
DROP DATABASE hello_world_db;

CREATE DATABASE dog_walking_app;
USE dog_walking_app;
SELECT dog_walking_app;
```

Table, variable types:

* Int

* varchar (a variable-length string between 1-255 characters)

  varchar(100)

```sql
CREATE DATABASE cat_app;
CREATE TABLE cats
	(
    name VARCHAR(100),
    age INT
	);
	
SHOW TABLES;
SHOW COLUMNS FROM cats; # same
DESC cats; # same

DROP TABLE cats;
```

```sql
CREATE TABLE pastries (name VARCHAR(50), age INT);
SHOW TABLES;
DESC patries;
DROP TABLE patries;
SHOW TABLES;
```

## 2. Inserting Data

```sql
INSERT INTO cats(name, age) VALUES ('Jetson', 7);

INSERT INTO cats(age, name) VALUES (7,'Jetson');

# preferable way:
INSERT INTO cats
						(NAME, 
             age) 
VALUES 			('Jetson', 
             7);
```

```sql
CREATE TABLE cats(name VARCHAR(50), age INT);
INSERT INTO cats(name, age) VALUES ('Blue', 1);
INSERT INTO cats(age, name) VALUES (7,'Jetson');
INSERT INTO cats(name, age) VALUES ('Amy', 1), ('Sadie', 3), ('Lazy', 1);

SELECT * FROM cats;
```

```sql
CREATE TABLE people(first_name VARCHAR(20), last_name VARCHAR(20), age INT);
DESC people;
INSERT INTO people(first_name, last_name, age) VALUES ("Tina", "Belcher", 13);
INSERT INTO people(last_name, age,first_name) VALUES ("Belcher", 13,"Tina");
INSERT INTO people(first_name, last_name, age) VALUES ("Linda", "Belcher", 45), ("Phillip", "Frond", 38), ("Calvin", "Fischoeder", 70);

SELECT * FROM people;
DROP TABLE people;
SHOW TABLES;
```

```sql
# set sql_mode='';
SHOW WARNINGS;
```

NULL: The value is unknown. 

Null = YES: they have an unknown value that happens by default and permitted to be null.

```sql
INSERT INTO cats(name) VALUES ('Jetson');
INSERT INTO cats() VALUES ();
```

```sql
CREATE TABLE cats2 
	(
    name VARCHAR(100) NOT NULL, 
   	age INT NOT NULL
  );
DESC cats2;
INSERT INTO cats2() VALUES (); 
# empty string and 0 integer.
```

Default values:

```sql
CREATE TABLE cats3 
	(
    name VARCHAR(100) DEFAULT "unnamed", 
    age INT DEFAULT 99
  );
  
INSERT INTO cats3() VALUES();
INSERT INTO cats3(NAME) VALUES(NULL);

CREATE TABLE cats3 
	(
    name VARCHAR(100) NOT NULL DEFAULT "unnamed", 
    age INT NOT NULL DEFAULT 99
  );
# This does not allow NULL.  

```

Primary key: a unique identifier

```sql
CREATE TABLE unique_cats 
	(
    cat_id INT NOT NULL, 
   	name VARCHAR(100), 
    age INT, 
    PRIMARY KEY (cat_id)
  );

DESC unique_cats;
INSERT INTO unique_cats(cat_id, name, age) VALUES(1, "Fred", 23);
SELECT * FROM unique_cats;
INSERT INTO unique_cats(cat_id, name, age) VALUES(2, "Louise", 3);
```

```sql
CREATE TABLE unique_cats2 
	(
    cat_id INT NOT NULL AUTO_INCREMENT, 
   	name VARCHAR(100), 
    age INT, 
    PRIMARY KEY (cat_id)
  );
INSERT INTO unique_cats2(name, age) VALUES("Butter", 4), ("Jiff", 2), ("Jiff", 2), ("Jiff", 2);
```

```sql
CREATE TABLE employees (
	id INT NOT NULL AUTO_INCREMENT,
  last_name VARCHAR(255) NOT NULL,
  first_name VARCHAR(255) NOT NULL,
  middel_name VARCHAR(255),
  age IN T NOT NULL,
  current_status VARCHAR(100) NOT NULL DEFAULT "employed",
  PRIMARY KEY (id)
);

# equivalently
CREATE TABLE employees (
	id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  last_name VARCHAR(255) NOT NULL,
  first_name VARCHAR(255) NOT NULL,
  middel_name VARCHAR(255),
  age IN T NOT NULL,
  current_status VARCHAR(100) NOT NULL DEFAULT "employed"
);
```

## 3. CRUD Command

Create, Read, Update, Delete.

```sql
SELECT name FROM cats;
SELECT name, age FROM cats;

SELECT * FROM cats WHERE age=4;
SELECT * FROM cats WHERE name='Egg';
SELECT * FROM cats WHERE name='EGG'; # case insensitive
```

```sql
SELECT cat_id FROM cats; 
SELECT name, breed FROM cats; 
SELECT name, age FROM cats WHERE breed='Tabby'; 
SELECT cat_id, age FROM cats WHERE cat_id=age; 
SELECT * FROM cats WHERE cat_id=age; 
```

Alias

```sql
SELECT cat_id AS id, name FROM cats;
SELECT name AS 'cat name', breed AS 'kitty breed' FROM cats;
DESC cats;
```

Update

```sql
UPDATE cats SET breed='Shorthair' WHERE breed='Tabby';
UPDATE cats SET age=14 WHERE name='Misty'; 
```

```sql
SELECT * FROM cats WHERE name='Jackson';
UPDATE cats SET name='Jack' WHERE name='Jackson';
SELECT * FROM cats WHERE name='Jackson';
SELECT * FROM cats WHERE name='Jack';
SELECT * FROM cats WHERE name='Ringo';
UPDATE cats SET breed='British Shorthair' WHERE name='Ringo';
SELECT * FROM cats WHERE name='Ringo';
SELECT * FROM cats;
SELECT * FROM cats WHERE breed='Maine Coon';
UPDATE cats SET age=12 WHERE breed='Maine Coon';
SELECT * FROM cats WHERE breed='Maine Coon';
```

 Delete:

```sql
DELETE FROM cats WHERE name='Egg';
SELECT * FROM cats;
SELECT * FROM cats WHERE name='egg';
DELETE FROM cats WHERE name='egg';
SELECT * FROM cats;
DELETE FROM cats; # still has the table but entries deleted
```

## 4. CRUD Challenge Section

```sql
SELECT database(); # check used database
CREATE DATABASE shirts_db;
USE shirts_db;
SELECT database(); # check used database

CREATE TABLE shirts 
	(
    shirt_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, 
  	article VARCHAR(100),
    color VARCHAR(100),
    shirt_size VARCHAR(100),
    last_worn INT
  );
  
DESC shirts;
SELECT * FROM shirts;

INSERT INTO shirts(article, color, shirt_size, last_worn) VALUES("t-shirt","white", "S", 10), ("t-shirt", "green", "S", 200), ("polo shirt", 'black', 'M', 10), ("tank top", "blue", "S", 50), ("t-shirt", "pink", "S", 0), ("polo shirt", "pink", "S", 0), ("polo shirt", "red", "M", 5), ("tank top", "white", "S", 200), ("tank top", "blue", "M", 15);
INSERT INTO shirts(color, article, shirt_size, last_worn) VALUES("purple", "polo shirt", "M", 50)

SELECT article, color FROM shirts;
SELECT article, color, shirt_size, last_worn FROM shirts WHERE shirt_size = "M";

UPDATE shirts SET shirt_size="L" WHERE article="polo shirt";
UPDATE shirts SET last_worn=0 WHERE last_worn=15;
UPDATE shirts SET color="off white", shirt_size="XS" WHERE color="white";

DELETE FROM shirts WHERE last_worn = 200;
DELETE FROM shirts WHERE article = "tank tops";

DELETE FROM shirts; # delete all entries but not the table
DROP TABLE shirts; # drop the entire shirts table
```

