rm -f *.sqlite *.nsys-rep
nsys profile --trace=cuda --gpu-metrics-device=all ./read --families 1 --requests 500 --prefix_mb 1800 --suffix_mb 3 --iters 1
nsys export --t sqlite -o report1.sqlite report1.nsys-rep
sqlite3 report1.sqlite "
SELECT 
    AVG(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) as avg_dram_read,
    MAX(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) as peak_dram_read
FROM GENERIC_EVENTS
WHERE json_extract(data, '$.\"DRAM Read Throughput\"') IS NOT NULL
  AND (CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL) > 0 
       OR CAST(json_extract(data, '$.\"DRAM Write Throughput\"') AS REAL) > 0);
" >> .out


rm -f *.sqlite *.nsys-rep
nsys profile --trace=cuda --gpu-metrics-device=all ./read --families 5 --requests 100 --prefix_mb 1800 --suffix_mb 3 --iters 1
nsys export --t sqlite -o report1.sqlite report1.nsys-rep
sqlite3 report1.sqlite "
SELECT 
    AVG(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) as avg_dram_read,
    MAX(CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL)) as peak_dram_read
FROM GENERIC_EVENTS
WHERE json_extract(data, '$.\"DRAM Read Throughput\"') IS NOT NULL
  AND (CAST(json_extract(data, '$.\"DRAM Read Throughput\"') AS REAL) > 0 
       OR CAST(json_extract(data, '$.\"DRAM Write Throughput\"') AS REAL) > 0);
" >> .out
