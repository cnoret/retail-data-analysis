-- Staging layer: typed views over the three raw CSV files.
-- Raw dates are DD/MM/YYYY and missing values are written as "NA".

CREATE OR REPLACE VIEW stores AS
SELECT
    Store::INTEGER AS store_id,
    Type           AS store_type,
    Size::INTEGER  AS size_sqft
FROM read_csv('data/stores.csv', header = true);

CREATE OR REPLACE VIEW sales AS
SELECT
    Store::INTEGER        AS store_id,
    Dept::INTEGER         AS dept_id,
    Date::DATE            AS week_date,
    Weekly_Sales::DOUBLE  AS weekly_sales,
    IsHoliday::BOOLEAN    AS is_holiday
FROM read_csv('data/sales.csv', header = true, dateformat = '%d/%m/%Y');

CREATE OR REPLACE VIEW features AS
SELECT
    Store::INTEGER               AS store_id,
    Date::DATE                   AS week_date,
    Temperature::DOUBLE          AS temperature,
    Fuel_Price::DOUBLE           AS fuel_price,
    MarkDown1::DOUBLE            AS markdown_1,
    MarkDown2::DOUBLE            AS markdown_2,
    MarkDown3::DOUBLE            AS markdown_3,
    MarkDown4::DOUBLE            AS markdown_4,
    MarkDown5::DOUBLE            AS markdown_5,
    CPI::DOUBLE                  AS cpi,
    Unemployment::DOUBLE         AS unemployment,
    IsHoliday::BOOLEAN           AS is_holiday
FROM read_csv('data/features.csv', header = true, dateformat = '%d/%m/%Y', nullstr = 'NA');
