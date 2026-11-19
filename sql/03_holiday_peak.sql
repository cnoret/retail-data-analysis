-- Q3. How big is the holiday peak?
-- Idea: total sales of each week, compared with a typical (median) week.

WITH weekly AS (
    SELECT
        week_date,
        SUM(weekly_sales) AS total_sales,
        MAX(CASE WHEN is_holiday THEN 1 ELSE 0 END) AS holiday_flag
    FROM sales
    GROUP BY week_date
)
SELECT
    week_date,
    holiday_flag,
    ROUND(total_sales / 1e6, 1) AS total_sales_m,
    ROUND(total_sales / (SELECT MEDIAN(total_sales) FROM weekly) - 1, 3) AS lift_vs_median_week
FROM weekly
ORDER BY total_sales DESC
LIMIT 10;
