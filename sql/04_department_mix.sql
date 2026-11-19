-- Q4. Which departments weigh the most, and which ones grow or shrink?
-- Growth compares February-October 2012 with February-October 2010.

SELECT
    dept_id,
    ROUND(SUM(weekly_sales) / (SELECT SUM(weekly_sales) FROM sales), 3) AS revenue_share,
    ROUND((
        SUM(CASE WHEN YEAR(week_date) = 2012 AND MONTH(week_date) BETWEEN 2 AND 10 THEN weekly_sales ELSE 0 END)
      - SUM(CASE WHEN YEAR(week_date) = 2010 AND MONTH(week_date) BETWEEN 2 AND 10 THEN weekly_sales ELSE 0 END)
    ) / 1e6, 1) AS change_2010_2012_m
FROM sales
GROUP BY dept_id
ORDER BY revenue_share DESC
LIMIT 15;
