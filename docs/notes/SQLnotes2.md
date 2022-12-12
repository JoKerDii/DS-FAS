# SQL

## 5. String Functions

```sql
DROP DATABASE IF EXISTS book_shop;
CREATE DATABASE book_shop;
USE book_shop; 

CREATE TABLE books 
	(
		book_id INT NOT NULL AUTO_INCREMENT,
		title VARCHAR(100),
		author_fname VARCHAR(100),
		author_lname VARCHAR(100),
		released_year INT,
		stock_quantity INT,
		pages INT,
		PRIMARY KEY(book_id)
	);

INSERT INTO books (title, author_fname, author_lname, released_year, stock_quantity, pages)
VALUES
('The Namesake', 'Jhumpa', 'Lahiri', 2003, 32, 291),
('Norse Mythology', 'Neil', 'Gaiman',2016, 43, 304),
('American Gods', 'Neil', 'Gaiman', 2001, 12, 465),
('Interpreter of Maladies', 'Jhumpa', 'Lahiri', 1996, 97, 198),
('A Hologram for the King: A Novel', 'Dave', 'Eggers', 2012, 154, 352),
('The Circle', 'Dave', 'Eggers', 2013, 26, 504),
('The Amazing Adventures of Kavalier & Clay', 'Michael', 'Chabon', 2000, 68, 634),
('Just Kids', 'Patti', 'Smith', 2010, 55, 304),
('A Heartbreaking Work of Staggering Genius', 'Dave', 'Eggers', 2001, 104, 437),
('Coraline', 'Neil', 'Gaiman', 2003, 100, 208),
('What We Talk About When We Talk About Love: Stories', 'Raymond', 'Carver', 1981, 23, 176),
("Where I'm Calling From: Selected Stories", 'Raymond', 'Carver', 1989, 12, 526),
('White Noise', 'Don', 'DeLillo', 1985, 49, 320),
('Cannery Row', 'John', 'Steinbeck', 1945, 95, 181),
('Oblivion: Stories', 'David', 'Foster Wallace', 2004, 172, 329),
('Consider the Lobster', 'David', 'Foster Wallace', 2005, 92, 343);
```

```sql
ls
mysql-ctl cli
USE cat_app;
source file_name.sql;
source testing/file_name.sql;
```

CONCAT: combine data for cleaner output.

```sql
DESC books;
SELECT * FROM books; 

SELECT CONCAT(author_fname, ' ', author_lname) FROM books;

SELECT
  CONCAT(author_fname, ' ', author_lname)
  AS 'full name'
FROM books;

SELECT 
	author_fname AS first, author_lname AS last, 
	CONCAT(author_fname, ', ', author_lname) AS full 
FROM books;
```

CONCAT_WS: concat with separator

```sql
SELECT CONCAT(title, '-', author_fname, '-', author_lname) FROM books;

SELECT 
    CONCAT_WS(' - ', title, author_fname, author_lname) 
FROM books;
```

SUBSTRING

SUBSTR

```sql
SELECT SUBSTRING('Hello World', 1, 4); # Hell
SELECT SUBSTRING('Hello World', 7); # World
SELECT SUBSTRING('Hello World', -3); # rld

SELECT title FROM books;
SELECT SUBSTRING("Where I'm Calling From: Selected Stories", 1, 10); # Use double quots!

SELECT SUBSTRING(title, 1, 10) FROM books;
SELECT SUBSTRING(title, 1, 10) AS 'short title' FROM books;

SELECT SUBSTR(title, 1, 10) AS 'short title' FROM books;
```

```sql
SELECT CONCAT
    (
        SUBSTRING(title, 1, 10),
        '...'
    )
FROM books;

source book_code.sql

SELECT CONCAT
    (
        SUBSTRING(title, 1, 10),
        '...'
    ) AS 'short title'
FROM books;

source book_code.sql
```

REPLACE: replace parts of strings

```sql
SELECT REPLACE('Hello World', 'Hell', '%$#@');
SELECT REPLACE('Hello World', 'l', '7');
SELECT REPLACE('Hello WOrld', 'o', '0'); # case sensitive!

SELECT
  REPLACE('cheese bread coffee milk', ' ', ' and ');

SELECT REPLACE(title, 'e ', '3') FROM books;

-- SELECT
--    CONCAT
--    (
--        SUBSTRING(title, 1, 10),
--        '...'
--    ) AS 'short title'
-- FROM books;

SELECT
    SUBSTRING(REPLACE(title, 'e', '3'), 1, 10)
FROM books;

SELECT
    SUBSTRING
    (
      REPLACE(title, 'e', '3'), 1, 10
    ) AS 'weird string'
FROM books;
```

REVERSE: 

```sql
SELECT REVERSE('Hello World');
SELECT REVERSE('meow meow');
SELECT REVERSE(author_fname) FROM books;
SELECT CONCAT('woof', REVERSE('woof'));
SELECT CONCAT(author_fname, REVERSE(author_fname)) FROM books;
```

CHAR_LENGTH:

```sql
SELECT CHAR_LENGTH('Hello World');

SELECT author_lname, CHAR_LENGTH(author_lname) AS 'length' FROM books;

SELECT CONCAT(author_lname, ' is ', CHAR_LENGTH(author_lname), ' characters long') FROM books;
```

UPPER(), LOWER()

```sql
SELECT UPPER('Hello World');
SELECT LOWER('Hello World');
SELECT UPPER(title) FROM books;
SELECT CONCAT('MY FAVORITE BOOK IS ', UPPER(title)) FROM books;
SELECT CONCAT('MY FAVORITE BOOK IS ', LOWER(title)) FROM books;
```

```sql
SELECT UPPER(REVERSE("Why does my cat look at me with such hatred?"));

SELECT REPLACE(title, " ", "->") AS title FROM books;

SELECT author_fname AS "forward", REVERSE(author_lname) AS "backward" FROM books;

SELECT CONCAT(author_fname, " ", author_lname) AS "full name in caps" FROM books;

SELECT CONCAT(title, " was released in ", released_year) AS "blurb" FROM books;

SELECT title, CHAR_LENGTH(title) AS "character length" FROM books;

SELECT CONCAT(SUBSTRING(title, 1,10), "...") AS "short title", CONCAT(author_lname,"," author_fname) AS "author", CONCAT(stock_quantity, " in stock") AS "quantity" FROM books;
```

## 6. Refining Selections

```sql
INSERT INTO books
    (title, author_fname, author_lname, released_year, stock_quantity, pages)
    VALUES ('10% Happier', 'Dan', 'Harris', 2014, 29, 256), 
           ('fake_book', 'Freida', 'Harris', 2001, 287, 428),
           ('Lincoln In The Bardo', 'George', 'Saunders', 2017, 1000, 367);

SELECT title FROM books;
```

ORDER BY: sorting our results.

```sql
SELECT author_lname FROM books ORDER BY author_lname; # ascending by default
SELECT title FROM books ORDER BY title;

SELECT author_lname FROM books ORDER BY author_lname DESC;
SELECT released_year FROM books ORDER BY released_year DESC;
SELECT released_year FROM books ORDER BY released_year ASC;

SELECT title, released_year, pages FROM books ORDER BY released_year;
SELECT title, pages FROM books ORDER BY released_year;

SELECT title, author_fname, author_lname 
FROM books ORDER BY 2; # author_fname
SELECT title, author_fname, author_lname 
FROM books ORDER BY 3; # author_lname
SELECT title, author_fname, author_lname 
FROM books ORDER BY 1 DESC; # title

SELECT author_fname, author_lname FROM books 
ORDER BY author_lname, author_fname;
```

LIMIT

```sql
SELECT title FROM books LIMIT 3;
SELECT * FROM books LIMIT 1;

SELECT title, released_year FROM books 
ORDER BY released_year DESC LIMIT 5; # first 5 books

SELECT title, released_year FROM books 
ORDER BY released_year DESC LIMIT 0,5; # same, first 5 books

SELECT title, released_year FROM books 
ORDER BY released_year DESC LIMIT 10,1; # the 11th book (from 10, 1 more)

SELECT title, released_year FROM books 
ORDER BY released_year DESC LIMIT 10,10; # the 11th to 20th book (from 10, 10 more)

SELECT * FROM tbl LIMIT 95,18446744073709551615; # use gigantic number to specific the end of the table

SELECT title FROM books LIMIT 5, 123219476457;
 
```

LIKE

```sql
SELECT title, author_fname FROM books WHERE author_fname LIKE '%da%';
SELECT title, author_fname FROM books WHERE author_fname LIKE 'da%';

SELECT title FROM books WHERE title LIKE 'the';
SELECT title FROM books WHERE title LIKE '%the';
SELECT title FROM books WHERE title LIKE '%the%';
```

```sql
SELECT title, stock_quantity FROM books;
SELECT title, stock_quantity FROM books WHERE stock_quantity LIKE '____'; # 4 digits

SELECT title, stock_quantity FROM books WHERE stock_quantity LIKE '__'; # 2 digits

(235)234-0987 LIKE '(___)___-____'

SELECT title FROM books WHERE title LIKE '%\%%';
SELECT title FROM books WHERE title LIKE '%\_%';
```

 ```sql
 SELECT title FROM books WHERE title LIKE '%stories%';
 
 SELECT title, pages FROM books ORDER BY pages DESC LIMIT 1;
 
 SELECT CONCAT(title, " - ", released_year) AS "summary" FROM books ORDER BY releaased_year DESC LIMIT 3;
 
 SELECT title, author_lname FROM books WHERE author_lname LIKE '% %';
 
 SELECT title, released_year, stock_quantity FROM books ORDER BY stock_quantity LIMIT 3;
 
 SELECT title, author_lname FROM books ORDER BY 2,1;
 
 SELECT CONCAT("MY FAVORITE AUTHOR IS ", UPPER(author_fname), " ", UPPER(author_lname),  "!") AS "yell" FROM books ORDER BY author_lname;
 ```

## 7. Aggregate Functions

COUNT

```sql
SELECT COUNT(*) FROM books;
SELECT COUNT(author_fname) FROM books;

SELECT COUNT(DISTINCT author_fname) FROM books;
SELECT COUNT(DISTINCT author_lname, author_fname) FROM books;

SELECT title FROM books WHERE title LIKE '%the%';
SELECT COUNT(*) FROM books WHERE title LIKE '%the%';
```

GROUP BY

```sql
SELECT title, author_lname FROM books;

SELECT title, author_lname FROM books
GROUP BY author_lname;

SELECT author_lname, COUNT(*) 
FROM books GROUP BY author_lname;

SELECT title, author_fname, author_lname FROM books;
SELECT title, author_fname, author_lname FROM books GROUP BY author_lname;

SELECT author_fname, author_lname, COUNT(*) FROM books GROUP BY author_lname;
SELECT author_fname, author_lname, COUNT(*) FROM books GROUP BY author_lname, author_fname;

SELECT released_year FROM books;
SELECT released_year, COUNT(*) FROM books GROUP BY released_year;

SELECT CONCAT('In ', released_year, ' ', COUNT(*), ' book(s) released') AS year FROM books GROUP BY released_year;
```

MIN and MAX

```sql
SELECT MIN(released_year) 
FROM books;

SELECT MAX(pages) 
FROM books;

SELECT MAX(pages), title
FROM books;
```

Sub query

```sql
SELECT * FROM books 
WHERE pages = (SELECT Min(pages) 
                FROM books); 

SELECT title, pages FROM books 
WHERE pages = (SELECT Max(pages) 
                FROM books); 

SELECT title, pages FROM books 
WHERE pages = (SELECT Min(pages) 
                FROM books); 

SELECT * FROM books 
ORDER BY pages ASC LIMIT 1;

SELECT title, pages FROM books 
ORDER BY pages ASC LIMIT 1;

SELECT * FROM books 
ORDER BY pages DESC LIMIT 1;
```

MIN/MAX and GROUPBY

```sql
SELECT author_fname, 
       author_lname, 
       Min(released_year) 
FROM   books 
GROUP  BY author_lname, 
          author_fname;

SELECT
  author_fname,
  author_lname,
  Max(pages)
FROM books
GROUP BY author_lname,
         author_fname;

SELECT
  CONCAT(author_fname, ' ', author_lname) AS author,
  MAX(pages) AS 'longest book'
FROM books
GROUP BY author_lname,
         author_fname;
```

SUM

```sql
SELECT SUM(pages)
FROM books;

SELECT SUM(released_year) FROM books;

SELECT author_fname,
       author_lname,
       Sum(pages)
FROM books
GROUP BY
    author_lname,
    author_fname;

SELECT author_fname,
       author_lname,
       Sum(released_year)
FROM books
GROUP BY
    author_lname,
    author_fname;
```

AVG

```sql
SELECT AVG(released_year) 
FROM books;

SELECT AVG(pages) 
FROM books;

SELECT AVG(stock_quantity) 
FROM books 
GROUP BY released_year;

SELECT released_year, AVG(stock_quantity) 
FROM books 
GROUP BY released_year;

SELECT author_fname, author_lname, AVG(pages) FROM books
GROUP BY author_lname, author_fname;
```

Exercise

```sql
SELECT COUNT(*) FROM books;

SELECT COUNT(*) FROM books GROUP BY released_year;

SELECT released_year, COUNT(*) FROM books GROUP BY released_year;

SELECT Sum(stock_quantity) FROM BOOKS;

SELECT AVG(released_year) FROM books GROUP BY author_lname, author_fname;

SELECT author_fname, author_lname, AVG(released_year) FROM books GROUP BY author_lname, author_fname;

SELECT CONCAT(author_fname, ' ', author_lname) FROM books
WHERE pages = (SELECT Max(pages) FROM books);

SELECT CONCAT(author_fname, ' ', author_lname) FROM books
ORDER BY pages DESC LIMIT 1;

SELECT pages, CONCAT(author_fname, ' ', author_lname) FROM books
ORDER BY pages DESC;

SELECT released_year AS year,
    COUNT(*) AS '# of books',
    AVG(pages) AS 'avg pages'
FROM books
    GROUP BY released_year;
```

## 8. Data Types

**Storying Text**

* VARCHART: 
* CHAR: fixed length, faster 

```sql
CREATE TABLE dogs (name CHAR(5), breed VARCHAR(10));
INSERT INTO dogs (name, breed) VALUES ('bob', 'beagle');

INSERT INTO dogs (name, breed) VALUES ('Princess Jane', 'Retrievesadfdsafdasfsafr');
SELECT * FROM dogs;
```

**Numbers**

* INT: whole numbers
* DECIMAL(13,2):

```sql
CREATE TABLE items(price DECIMAL(5,2)); # five digits with two decimal places
INSERT INTO items(price) VALUES(7);
INSERT INTO items(price) VALUES(7987654);
INSERT INTO items(price) VALUES(34.88);
SELECT * FROM items;
```

**FLOAT and DOUBLE**

* Memory needed: 4 Bytes and 8 Bytes

* Precision Issues: ~ 7 digits and ~ 15 digits

```sql
CREATE TABLE thingies (price FLOAT);
INSERT INTO thingies(price) VALUES (88.45);
SELECT * FROM thingies;
INSERT INTO thingies(price) VALUES (8877665544.45);
SELECT * FROM thingies;
```

**DATE**

* Values with a Date but not time
* 'YYYY-MM-DD' format

**TIME**

* Values with a time but no Date
* 'HH:MM:SS' format

**DATETIME (most used)**

* Values with a date AND Time

* 'YYYY-MM-DD HH:MM:SS' format

```sql
CREATE TABLE people (name VARCHAR(100), birthdate DATE, birthtime TIME, birthdt DATETIME);
INSERT INTO people (name, birthdate, birthtime, birthdt)
VALUES('Padma', '1983-11-11', '10:07:35', '1983-11-11 10:07:35');
INSERT INTO people (name, birthdate, birthtime, birthdt)
VALUES('Larry', '1943-12-25', '04:10:42', '1943-12-25 04:10:42');
SELECT * FROM people;
```

CURDATE() gives current date, 

CURTIME() gives current time, 

NOW() gives current datetime

```sql
INSERT INTO people (name, birthdate, birthtime, birthdt) 
VALUES ('Microwave', CURDATE(), CURTIME(), NOW())
DELETE FROM people WHERE name = 'Microwave'
```

**Formatting DATES**

```sql
SELECT name, birthdate FROM people;
SELECT name, DAY(birthdate) FROM people;
SELECT name, birthdate, DAY(birthdate) FROM people;
SELECT name, birthdate, DAYNAME(birthdate) FROM people;
SELECT name, birthdate, DAYOFWEEK(birthdate) FROM people;
SELECT name, birthdate, DAYOFYEAR(birthdate) FROM people;
SELECT name, birthtime, DAYOFYEAR(birthtime) FROM people;
SELECT name, birthdt, DAYOFYEAR(birthdt) FROM people;
SELECT name, birthdt, MONTH(birthdt) FROM people;
SELECT name, birthdt, MONTHNAME(birthdt) FROM people;
SELECT name, birthtime, HOUR(birthtime) FROM people;
SELECT name, birthtime, MINUTE(birthtime) FROM people;

SELECT CONCAT(MONTHNAME(birthdate), ' ', DAY(birthdate), ' ', YEAR(birthdate)) FROM people;

SELECT DATA_FORMAT('2009-10-04 22:23:00', '%W%M%Y');
SELECT DATE_FORMAT(birthdt, 'Was born on a %W') FROM people;
SELECT DATE_FORMAT(birthdt, '%m/%d/%Y') FROM people;
SELECT DATE_FORMAT(birthdt, '%m/%d/%Y at %h:%i') FROM people;
```

**DATE MATH**

* Function: DATEDIFF

```sql
SELECT * FROM people;
SELECT DATEDIFF(NOW(), birthdate) FROM people;
SELECT name, birthdate, DATEDIFF(NOW(), birthdate) FROM people;

SELECT birthdt FROM people;
SELECT birthdt, DATE_ADD(birthdt, INTERVAL 1 MONTH) FROM people;
SELECT birthdt, DATE_ADD(birthdt, INTERVAL 10 SECOND) FROM people;
SELECT birthdt, DATE_ADD(birthdt, INTERVAL 3 QUARTER) FROM people;

SELECT birthdt, birthdt + INTERVAL 1 MONTH FROM people;
SELECT birthdt, birthdt - INTERVAL 5 MONTH FROM people;
SELECT birthdt, birthdt + INTERVAL 15 MONTH + INTERVAL 10 HOUR FROM people;
```

**TIMESTAMPS**

Can only store date and time from '1970-01-01 00:00:01' to '2039-01-19 03:14:07'.

While DATETIME can store date and time from '1000-01-01 00:00:00' to '9999-12-31 23:59:59'.

```sql
CREATE TABLE comments (
    content VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO comments (content) VALUES('lol what a funny article');
INSERT INTO comments (content) VALUES('I found this offensive');
INSERT INTO comments (content) VALUES('Ifasfsadfsadfsad');
SELECT * FROM comments ORDER BY created_at DESC;

CREATE TABLE comments2 (
    content VARCHAR(100),
    changed_at TIMESTAMP DEFAULT NOW() ON UPDATE CURRENT_TIMESTAMP
);

INSERT INTO comments2 (content) VALUES('dasdasdasd');
INSERT INTO comments2 (content) VALUES('lololololo');
INSERT INTO comments2 (content) VALUES('I LIKE CATS AND DOGS');
UPDATE comments2 SET content='THIS IS NOT GIBBERISH' WHERE content='dasdasdasd';
SELECT * FROM comments2;
SELECT * FROM comments2 ORDER BY changed_at;

CREATE TABLE comments2 (
    content VARCHAR(100),
    changed_at TIMESTAMP DEFAULT NOW() ON UPDATE NOW()
);
```

**Exercise:**

What's a good use case for CHAR?

Used for text that we know has a fixed length, e.g., State abbreviations, abbreviated company names, sex M/F, etc.

```sql
CREATE TABLE inventory (
    item_name VARCHAR(100),
    price DECIMAL(8,2),
    quantity INT
);
```

What's the difference between DATETIME and TIMESTAMP?

They both store datetime information, but there's a difference in the range,  TIMESTAMP has a smaller range. TIMESTAMP also takes up less space.  TIMESTAMP is used for things like meta-data about when something is created or updated.

```sql
SELECT CURTIME();
SELECT CURDATE();

SELECT DAYOFWEEK(CURDATE());
SELECT DAYOFWEEK(NOW());
SELECT DATE_FORMAT(NOW(), '%w') + 1;

SELECT DAYNAME(NOW());
SELECT DATE_FORMAT(NOW(), '%W');

SELECT DATE_FORMAT(CURDATE(), '%m/%d/%Y');

SELECT DATE_FORMAT(NOW(), '%M %D at %h:%i');

CREATE TABLE tweets(
    content VARCHAR(140),
    username VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO tweets (content, username) VALUES('this is my first tweet', 'coltscat');
SELECT * FROM tweets;

INSERT INTO tweets (content, username) VALUES('this is my second tweet', 'coltscat');
SELECT * FROM tweets;
```
