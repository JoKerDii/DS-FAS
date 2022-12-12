# SQL

## 9. Logical Operators

**Not equal**

```sql
SELECT title FROM books WHERE released_year = 2017;
SELECT title FROM books WHERE released_year != 2017;

SELECT title, author_lname FROM books WHERE author_lname = 'Harris';
SELECT title, author_lname FROM books WHERE author_lname != 'Harris';
```

**Not like**

```sql
SELECT title FROM books WHERE title LIKE 'W';
SELECT title FROM books WHERE title LIKE 'W%';
SELECT title FROM books WHERE title LIKE '%W%';
SELECT title FROM books WHERE title NOT LIKE 'W%';
```

**Greater than**

```sql
SELECT title, released_year FROM books ORDER BY released_year;

SELECT title, released_year FROM books 
WHERE released_year > 2000 ORDER BY released_year;

SELECT title, released_year FROM books 
WHERE released_year >= 2000 ORDER BY released_year;

SELECT 99 > 1;
SELECT 99 > 567;

SELECT title, author_lname FROM books WHERE author_lname = 'Eggers';
SELECT title, author_lname FROM books WHERE author_lname = 'eggers';
SELECT title, author_lname FROM books WHERE author_lname = 'eGGers';
```

**Less than**

```sql
SELECT title, released_year FROM books
WHERE released_year < 2000;
SELECT title, released_year FROM books
WHERE released_year <= 2000;
```

**AND (&& is deprecated in new MySQL)**

```sql
SELECT  
    title, 
    author_lname, 
    released_year FROM books
WHERE author_lname='Eggers' 
    AND released_year > 2010;

SELECT * 
FROM books
WHERE author_lname='Eggers' 
    AND released_year > 2010 
    AND title LIKE '%novel%';
```

**OR (|| is deprecated in new MySQL)**

```sql
SELECT 
    title, 
    author_lname, 
    released_year 
FROM books
WHERE author_lname='Eggers' || released_year > 2010;


SELECT title, 
       author_lname, 
       released_year, 
       stock_quantity 
FROM   books 
WHERE  author_lname = 'Eggers' 
              || released_year > 2010 
```

**(NOT) BETWEEN and CAST**

```sql
SELECT title, released_year FROM books 
WHERE released_year BETWEEN 2004 AND 2015;

SELECT title, released_year FROM books 
WHERE released_year NOT BETWEEN 2004 AND 2015;

SELECT CAST('2017-05-02' AS DATETIME);

show databases;
use new_testing_db;

SELECT name, birthdt FROM people WHERE birthdt BETWEEN '1980-01-01' AND '2000-01-01';

SELECT 
    name, 
    birthdt 
FROM people
WHERE 
    birthdt BETWEEN CAST('1980-01-01' AS DATETIME)
    AND CAST('2000-01-01' AS DATETIME);
```

**(NOT) IN**

```sql
show databases();
use book_shop;

SELECT title, author_lname FROM books
WHERE author_lname IN ('Carver', 'Lahiri', 'Smith');

SELECT title, released_year FROM books
WHERE released_year IN (2017, 1985);

SELECT title, released_year FROM books
WHERE released_year >= 2000
AND released_year NOT IN 
(2000,2002,2004,2006,2008,2010,2012,2014,2016);

SELECT title, released_year FROM books
WHERE released_year >= 2000 AND
released_year % 2 != 0;
```

**Case statements**

```sql
SELECT title, released_year,
       CASE 
         WHEN released_year >= 2000 THEN 'Modern Lit'
         ELSE '20th Century Lit'
       END AS GENRE
FROM books;

SELECT title, stock_quantity,
    CASE 
        WHEN stock_quantity BETWEEN 0 AND 50 THEN '*'
        WHEN stock_quantity BETWEEN 51 AND 100 THEN '**'
        WHEN stock_quantity BETWEEN 101 AND 150 THEN '***'
        ELSE '****'
    END AS STOCK
FROM books;
```

**Exercise**

```sql
SELECT title, released_year FROM books WHERE released_year < 1980;
SELECT title, author_lname FROM books WHERE author_lname='Eggers' OR author_lname='Chabon';
SELECT title, author_lname FROM books WHERE author_lname IN ('Eggers','Chabon');
SELECT title, author_lname, released_year FROM books WHERE author_lname = 'Lahiri' && released_year > 2000;
SELECT title, pages FROM books WHERE pages >= 100 && pages <=200;
SELECT title, pages FROM books WHERE pages BETWEEN 100 AND 200;

SELECT 
    title, 
    author_lname 
FROM books 
WHERE 
    author_lname LIKE 'C%' OR 
    author_lname LIKE 'S%';

SELECT 
    title, 
    author_lname 
FROM books 
WHERE 
    SUBSTR(author_lname,1,1) = 'C' OR 
    SUBSTR(author_lname,1,1) = 'S';

SELECT title, author_lname FROM books 
WHERE SUBSTR(author_lname,1,1) IN ('C', 'S');

SELECT 
    title, 
    author_lname,
    CASE
        WHEN title LIKE '%stories%' THEN 'Short Stories'
        WHEN title = 'Just Kids' OR title = 'A Heartbreaking Work of Staggering Genius' THEN 'Memoir'
        ELSE 'Novel'
    END AS TYPE
FROM books;

SELECT author_fname, author_lname,
    CASE 
        WHEN COUNT(*) = 1 THEN '1 book'
        ELSE CONCAT(COUNT(*), ' books')
    END AS COUNT
FROM books 
GROUP BY author_lname, author_fname;
```

## 10. One To Many

Creating the customers and orders tables

```sql
CREATE TABLE customers(
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(100)
);
CREATE TABLE orders(
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATE,
    amount DECIMAL(8,2),
    customer_id INT,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
);

INSERT INTO customers (first_name, last_name, email) 
VALUES ('Boy', 'George', 'george@gmail.com'),
       ('George', 'Michael', 'gm@gmail.com'),
       ('David', 'Bowie', 'david@gmail.com'),
       ('Blue', 'Steele', 'blue@gmail.com'),
       ('Bette', 'Davis', 'bette@aol.com');
       
INSERT INTO orders (order_date, amount, customer_id)
VALUES ('2016/02/10', 99.99, 1),
       ('2017/11/11', 35.50, 1),
       ('2014/12/12', 800.67, 2),
       ('2015/01/03', 12.50, 2),
       ('1999/04/11', 450.25, 5);
       
INSERT INTO orders (order_date, amount, customer_id)
VALUES ('2016/06/06', 33.67, 98);
```

Finding Orders placed by George in 2 steps:

```sql
SELECT id FROM customers WHERE last_name='George';
SELECT * FROM orders WHERE customer_id = 1;
```

Finding Orders placed by George using a subquery:

```sql
SELECT * FROM orders WHERE customer_id =
    (
        SELECT id FROM customers
        WHERE last_name='George'
    );
```

**Implicit Inner Join**

```sql
SELECT * FROM customers, orders WHERE customers.id = orders.customer_id;

SELECT first_name, last_name, order_date, amount
FROM customers, orders 
    WHERE customers.id = orders.customer_id;
```

**Explicit Inner Join** (recommended)

```sql
SELECT * FROM customers
JOIN orders
    ON customers.id = orders.customer_id;
    
SELECT first_name, last_name, order_date, amount 
FROM customers
JOIN orders
    ON customers.id = orders.customer_id;
    
SELECT *
FROM orders
JOIN customers
    ON customers.id = orders.customer_id;
    
SELECT first_name, last_name, order_date, amount 
FROM customers
JOIN orders
    ON customers.id = orders.customer_id
ORDER BY order_date;

SELECT 
    first_name, 
    last_name, 
    SUM(amount) AS total_spent
FROM customers
JOIN orders
    ON customers.id = orders.customer_id
GROUP BY orders.customer_id
ORDER BY total_spent DESC;
```

**Left Join**

IFNULL(variable, 0): check if "variable" is null, if it is then replace it with 0. 

```sql
SELECT * FROM customers
LEFT JOIN orders
    ON customers.id = orders.customer_id;

SELECT first_name, last_name, order_date, amount
FROM customers
LEFT JOIN orders
    ON customers.id = orders.customer_id; 

SELECT 
    first_name, 
    last_name,
    IFNULL(SUM(amount), 0) AS total_spent
FROM customers
LEFT JOIN orders
    ON customers.id = orders.customer_id
GROUP BY customers.id
ORDER BY total_spent;
```

**Right Join**

```sql
SELECT * FROM customers
RIGHT JOIN orders
    ON customers.id = orders.customer_id;
```

Advanced left join

```sql
SELECT 
    IFNULL(first_name,'MISSING') AS first, 
    IFNULL(last_name,'USER') AS last, 
    order_date, 
    amount, 
    SUM(amount)
FROM customers
RIGHT JOIN orders
    ON customers.id = orders.customer_id
GROUP BY first_name, last_name;
```

**Delete CASCADE**

When a customer is deleted that has a corresponding order, delete the order as well.

```sql
CREATE TABLE customers(
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE orders(
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATE,
    amount DECIMAL(8,2),
    customer_id INT,
    FOREIGN KEY(customer_id) 
        REFERENCES customers(id)
        ON DELETE CASCADE
);

DELETE FROM customers WHERE email = 'george@gmail.com';
```

## 11. Many to Many

1. Join the Series table with the reviews table.

```sql
SELECT 
    title, 
    rating 
FROM series
JOIN reviews
    ON series.id = reviews.series_id;
```

2. Group by the series ID, find average rating and sort the average rating from lower to larger.

```sql
SELECT
    title,
    AVG(rating) as avg_rating
FROM series
JOIN reviews
    ON series.id = reviews.series_id
GROUP BY series.id
ORDER BY avg_rating;
```

3. Join the reviewers and reviews tables and show the rating for each reviewer.

```sql
SELECT
    first_name,
    last_name,
    rating
FROM reviewers
INNER JOIN reviews
    ON reviewers.id = reviews.reviewer_id;
    
SELECT
    first_name,
    last_name,
    rating
FROM reviews
INNER JOIN reviewers
    ON reviewers.id = reviews.reviewer_id;
```

4. Identify unreviewed series.

```sql
SELECT title AS unreviewed_series
FROM series
LEFT JOIN reviews
    ON series.id = reviews.series_id
WHERE rating IS NULL;
```

5. Grab genre and compute average rating.

```sql
SELECT genre, 
       Round(Avg(rating), 2) AS avg_rating 
FROM   series 
       INNER JOIN reviews 
               ON series.id = reviews.series_id 
GROUP  BY genre; 
```

6. Summary statistics: count, min, max, avg, status

```sql
SELECT first_name, 
       last_name, 
       Count(rating)                               AS COUNT, 
       Ifnull(Min(rating), 0)                      AS MIN, 
       Ifnull(Max(rating), 0)                      AS MAX, 
       Round(Ifnull(Avg(rating), 0), 2)            AS AVG, 
       IF(Count(rating) > 0, 'ACTIVE', 'INACTIVE') AS STATUS 
FROM   reviewers 
       LEFT JOIN reviews 
              ON reviewers.id = reviews.reviewer_id 
GROUP  BY reviewers.id; 
```

With POWER USERS 

```sql
SELECT first_name, 
       last_name, 
       Count(rating)                    AS COUNT, 
       Ifnull(Min(rating), 0)           AS MIN, 
       Ifnull(Max(rating), 0)           AS MAX, 
       Round(Ifnull(Avg(rating), 0), 2) AS AVG, 
       CASE 
         WHEN Count(rating) >= 10 THEN 'POWER USER' 
         WHEN Count(rating) > 0 THEN 'ACTIVE' 
         ELSE 'INACTIVE' 
       end                              AS STATUS 
FROM   reviewers 
       LEFT JOIN reviews 
              ON reviewers.id = reviews.reviewer_id 
GROUP  BY reviewers.id; 
```

7. Put title, rating, and reviewer together.

```sql
SELECT 
    title,
    rating,
    CONCAT(first_name,' ', last_name) AS reviewer
FROM reviewers
INNER JOIN reviews 
    ON reviewers.id = reviews.reviewer_id
INNER JOIN series
    ON series.id = reviews.series_id
ORDER BY title;
```

