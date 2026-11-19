-- Q1. Which stores underperform, once their size is taken into account?
-- Idea: compare each store's sales per square foot with the median of stores of the same type.

WITH store_sales AS (
    SELECT
        s.store_id,
        st.store_type,
        st.size_sqft,
        SUM(s.weekly_sales) / st.size_sqft AS sales_per_sqft
    FROM sales AS s
    JOIN stores AS st ON st.store_id = s.store_id
    GROUP BY s.store_id, st.store_type, st.size_sqft
)
SELECT
    store_id,
    store_type,
    ROUND(sales_per_sqft, 0) AS sales_per_sqft,
    ROUND(MEDIAN(sales_per_sqft) OVER (PARTITION BY store_type), 0) AS type_median,
    ROUND(sales_per_sqft / MEDIAN(sales_per_sqft) OVER (PARTITION BY store_type) - 1, 3) AS gap_vs_type
FROM store_sales
ORDER BY gap_vs_type;
