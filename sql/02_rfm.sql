CREATE OR REPLACE TABLE rfm AS
WITH history AS (
    SELECT *
    FROM clean_transactions
    WHERE invoice_date < TIMESTAMP '2011-09-10'
),
base AS (
    SELECT
        customer_id,
        date_diff('day', MAX(invoice_date), TIMESTAMP '2011-09-10') AS recency_days,
        COUNT(DISTINCT invoice)                                     AS frequency,
        ROUND(SUM(line_total), 2)                                   AS monetary
    FROM history
    GROUP BY customer_id
),
ranked AS (
    SELECT
        *,
        PERCENT_RANK() OVER (ORDER BY recency_days DESC) AS r_pct,
        PERCENT_RANK() OVER (ORDER BY frequency ASC)     AS f_pct,
        PERCENT_RANK() OVER (ORDER BY monetary ASC)      AS m_pct
    FROM base
)
SELECT
    customer_id,
    recency_days,
    frequency,
    monetary,
    -- 5 = best. Tied values always get the same score
    LEAST(5, 1 + CAST(FLOOR(r_pct * 5) AS INTEGER)) AS r_score,
    LEAST(5, 1 + CAST(FLOOR(f_pct * 5) AS INTEGER)) AS f_score,
    LEAST(5, 1 + CAST(FLOOR(m_pct * 5) AS INTEGER)) AS m_score
FROM ranked;