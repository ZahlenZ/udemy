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