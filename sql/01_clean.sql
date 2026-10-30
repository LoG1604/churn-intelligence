CREATE OR REPLACE TABLE clean_transactions AS
SELECT DISTINCT
    invoice,
    stock_code,
    description,
    quantity,
    invoice_date,
    price,
    customer_id,
    country,
    ROUND(quantity * price, 2) AS line_total
FROM raw_transactions
WHERE customer_id IS NOT NULL                    -- can't track churn without a customer
  AND invoice NOT LIKE 'C%'                      -- cancellations
  AND quantity > 0
  AND price > 0
  AND regexp_matches(stock_code, '^[0-9]{5}');   -- real products only, drops POST, D, M, BANK CHARGES etc.