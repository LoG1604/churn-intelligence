CREATE OR REPLACE TABLE features AS
WITH history AS (
    SELECT *
    FROM clean_transactions
    WHERE invoice_date < TIMESTAMP '2011-09-10'
),
orders AS (
    SELECT
        customer_id,
        invoice,
        MIN(invoice_date) AS order_date
    FROM history
    GROUP BY customer_id, invoice
),
gaps AS (
    SELECT
        customer_id,
        date_diff(
            'day',
            LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
            order_date
        ) AS gap_days
    FROM orders
),
gap_avg AS (
    SELECT customer_id, AVG(gap_days) AS avg_gap_days
    FROM gaps
    GROUP BY customer_id
),
extra AS (
    SELECT
        customer_id,
        COUNT(DISTINCT stock_code) AS n_products,
        SUM(quantity)              AS total_items,
        date_diff('day', MIN(invoice_date), TIMESTAMP '2011-09-10') AS tenure_days,
        COUNT(DISTINCT CASE
            WHEN invoice_date >= TIMESTAMP '2011-09-10' - INTERVAL 90 DAY
            THEN invoice END)      AS orders_last_90d,
        MAX(CASE WHEN country = 'United Kingdom' THEN 1 ELSE 0 END) AS is_uk
    FROM history
    GROUP BY customer_id
),
future AS (
    SELECT DISTINCT customer_id
    FROM clean_transactions
    WHERE invoice_date >= TIMESTAMP '2011-09-10'
      AND invoice_date <  TIMESTAMP '2011-12-10'
)
SELECT
    r.customer_id,
    r.recency_days,
    r.frequency,
    r.monetary,
    r.r_score,
    r.f_score,
    r.m_score,
    ROUND(r.monetary / r.frequency, 2) AS avg_order_value,
    e.n_products,
    e.total_items,
    e.tenure_days,
    e.orders_last_90d,
    e.is_uk,
    g.avg_gap_days,                      -- NULL for one-order customers, XGBoost handles it
    CASE WHEN f.customer_id IS NULL THEN 1 ELSE 0 END AS churned
FROM rfm r
JOIN extra e         ON e.customer_id = r.customer_id
LEFT JOIN gap_avg g  ON g.customer_id = r.customer_id
LEFT JOIN future f   ON f.customer_id = r.customer_id;