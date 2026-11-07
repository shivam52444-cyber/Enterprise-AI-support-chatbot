-- Run ONCE in DBeaver (admin connection), after schema_lean.sql.
-- A human can now APPROVE a refund request, so the status needs the value 'approved',
-- and an approved request must still block a second request for the same item.

ALTER TABLE refund_requests DROP CONSTRAINT refund_requests_status_check;
ALTER TABLE refund_requests ADD CONSTRAINT refund_requests_status_check
    CHECK (status IN ('auto_approved', 'pending_human', 'approved', 'rejected'));

DROP INDEX one_active_refund_per_item;
CREATE UNIQUE INDEX one_active_refund_per_item
    ON refund_requests (order_id, order_item_id)
    WHERE status IN ('auto_approved', 'pending_human', 'approved');
