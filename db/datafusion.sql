CREATE TABLE values(
    a FLOAT,
    b STRING,
    c BIGINT
) AS VALUES
  (1.0, 'banana', 2),
  (2.0,  'apple', 3),
  (3.0, 'orange', 4),
  (NULL, 'banana', 2),
  (2.0,     NULL, 3),
  (3.0, 'orange', NULL);
;
CREATE TABLE structs AS SELECT named_struct('a', a,  'b', b, 'c', c) AS abc FROM values;