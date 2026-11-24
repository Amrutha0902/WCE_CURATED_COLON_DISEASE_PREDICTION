-- REVIEW BEFORE RUNNING ON AN EXISTING DEPLOYMENT.
-- Back up your MySQL database first.
-- The legacy predictions table may be missing its username column.
-- This migration is intended for MySQL 8.0 and runs only when the column is absent.
USE colon_disease_db;

SET @column_exists := (
  SELECT COUNT(*)
  FROM information_schema.columns
  WHERE table_schema = DATABASE() AND table_name = 'predictions'
    AND column_name = 'username'
);
SET @ddl := IF(@column_exists = 0,
    'ALTER TABLE predictions ADD COLUMN username VARCHAR(255) NULL',
    'SELECT ''predictions.username already exists''');
PREPARE migration_statement FROM @ddl;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

-- Legacy predictions have NULL username until accurately reassigned by an
-- authorized data steward. Do NOT assign all historical records to one user.
-- Users with plaintext passwords MUST complete a separately verified reset.
