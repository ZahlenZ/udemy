```SQL
SELECT
    UPPER(columns) AS alias
    ,LEFT(column, n)
    ,RIGHT(column, n)
    ,columns
FROM table
WHERE (stuff) AND (stuff BETWEEN this AND that)
GROUP BY column, column
HAVING SUM(column) > 100
ORDER BY column DESC, column ASC
```

concat string in MSSQL with + not with ||

in postgresql there is POSITION, in MSSQL it is CHARINDEX("string", column)

SUBSTRING(expression, start, length)

EXTRACT(field from column)
DAY
DOW
HOUR
MONTH
QUARTER
MONTH
WEEK
YEAR

IN MSSQL there is 
YEAR(date)
MONTH(date)
DAY(date)
HOUR(date)
MINUTE(date)
SECOND(date)

IN MSSQL there is FORMAT()
SELECT
    FORMAT(GETDATE(), 'dd/MM/yyyy') AS formatted_date

TO_CHAR(column, format)

CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    WHEN condition3 THEN result3
    WHEN conditionN THEN resultN
    ELSE result
END AS column_name


DATATYPES

NUMERIC
    - int
    - smallint
    - bigint
    - numeric (float)
    - serial (autoincrementing integer)
STRING
    - varchar
    - character
    - text
DATE/TIME
    - date
    - time (with/without timezone)
    - timestamp (date and time)
    - intervals
BOOLEAN
    - TRUE/FALSE/NULL
ENUM
    - list of ordered values
ARRAY
    - stores a list of values

CONTSTRAINTS
    - NOT NULL
    - UNIQUE
    - DEFAULT
    - PRIMARY KEY
    - REFERENCES (used for foreign keys)
    - CHECK (ensure values in column follow logic)

CREATE TABLE <table name> (
    column_name1 TYPE CONSTRAINT,
    column_name2 TYPE CONSTRAINT
)

order of values needs to match order of columns in table
INSERT INTO <table name> 
VALUES (value1, ..., valueN,)

INSERT INTO <table name>
(column1, ..., columnN)
(value1, ... , valueN)
(anotherrowvalue, ..., anotherrowvalueN)

ALTER <table name>
ALTER_ACTION

- DROP
- ADD
- TYPE
- ALTER <columnname> TYPE <newtype>
- RENAME


ALTER <table name>
ALTER ACTION row TYPE CONTSTRAINT

DROP <object> <name>

DROP TABLE<table name>

DROP SCHEMA <schema_name>

delete all of the data in the table
TRUNCATE <table name>


CREATE TABLE <table name> (
    <column name> TYPE CHECK(condition)
)

can add a name to contstraint by adding CONSTRAINT <'constraint name'>
default <table>_<column>_<constraint_check>
CREATE TABLE director (
    name TEXT CONSTRAINT name_length CHECK (length(name) > 1)
)
