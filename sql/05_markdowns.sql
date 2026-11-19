-- Q5. Do markdown promotions lift sales?
-- Markdown data starts in November 2011. Holiday weeks are left out (busy anyway).
-- For each store, weeks with above-average markdowns are compared with the other weeks.

WITH store_weeks AS (
    SELECT
        s.store_id,
        s.week_date,
        SUM(s.weekly_sales) AS store_sales,
        MAX(COALESCE(f.markdown_1, 0) + COALESCE(f.markdown_2, 0) + COALESCE(f.markdown_3, 0)
          + COALESCE(f.markdown_4, 0) + COALESCE(f.markdown_5, 0)) AS markdown
    FROM sales AS s
    JOIN features AS f ON f.store_id = s.store_id AND f.week_date = s.week_date
    WHERE s.week_date >= DATE '2011-11-11'
      AND NOT s.is_holiday
    GROUP BY s.store_id, s.week_date
),
compared AS (
    SELECT
        *,
        markdown > AVG(markdown) OVER (PARTITION BY store_id) AS high_markdown,
        store_sales / AVG(store_sales) OVER (PARTITION BY store_id) AS sales_vs_store_avg
    FROM store_weeks
)
SELECT
    CASE WHEN high_markdown THEN 'High markdown weeks' ELSE 'Other weeks' END AS week_type,
    COUNT(*) AS store_weeks,
    ROUND(AVG(sales_vs_store_avg) - 1, 3) AS sales_vs_store_avg
FROM compared
GROUP BY week_type;
