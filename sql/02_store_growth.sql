-- Q2. Which stores are growing, and which are declining?
-- The data stops in October 2012, so we compare the same months: February to October.

WITH yearly AS (
    SELECT
        store_id,
        SUM(CASE WHEN YEAR(week_date) = 2010 THEN weekly_sales ELSE 0 END) AS sales_2010,
        SUM(CASE WHEN YEAR(week_date) = 2012 THEN weekly_sales ELSE 0 END) AS sales_2012
    FROM sales
    WHERE MONTH(week_date) BETWEEN 2 AND 10
    GROUP BY store_id
)
SELECT
    store_id,
    ROUND(sales_2010 / 1e6, 1) AS sales_2010_m,
    ROUND(sales_2012 / 1e6, 1) AS sales_2012_m,
    ROUND(sales_2012 / sales_2010 - 1, 4) AS growth
FROM yearly
ORDER BY growth;
