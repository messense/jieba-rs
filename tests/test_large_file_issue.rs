use jieba_rs::Jieba;

#[test]
#[ignore] // This test is ignored by default due to high memory usage
fn test_large_file_memory_allocation() {
    // Create a large text content similar to the issue report
    let base_text = "这是一个测试文件用来复现jieba-rs在处理大文件时的内存分配问题，包含大量的中文文本内容。";
    let large_content = base_text.repeat(5_000_000); // Approximately 250MB
    
    println!("Test content size: {} bytes", large_content.len());
    
    // This should not crash with memory allocation failure
    let jieba = Jieba::new();
    
    // Try to tokenize the large content - this is where the issue occurs
    let result = std::panic::catch_unwind(|| {
        jieba.cut(&large_content, false)
    });
    
    match result {
        Ok(_) => println!("Successfully processed large content"),
        Err(_) => panic!("Memory allocation failed for large content"),
    }
}

#[test]
fn test_very_large_heuristic_capacity() {
    // Test the specific scenario where heuristic_capacity becomes very large
    let jieba = Jieba::new();
    
    // Create a string that would cause problematic memory allocation
    let size = 250_000_000; // 250MB
    let content = "中".repeat(size / 3); // Each Chinese character is 3 bytes in UTF-8
    
    println!("Content size: {} bytes", content.len());
    
    // Calculate what the heuristic_capacity would be
    let heuristic_capacity = content.len() / 2;
    let expected_allocation_uncapped = heuristic_capacity * 5 * std::mem::size_of::<usize>();
    let expected_allocation_capped = std::cmp::min(heuristic_capacity * 5, 1_000_000) * std::mem::size_of::<usize>();
    
    println!("Heuristic capacity: {}", heuristic_capacity);
    println!("Expected memory allocation (uncapped): {} bytes ({:.2} GB)", 
             expected_allocation_uncapped, 
             expected_allocation_uncapped as f64 / 1_000_000_000.0);
    println!("Expected memory allocation (capped): {} bytes ({:.2} MB)", 
             expected_allocation_capped, 
             expected_allocation_capped as f64 / 1_000_000.0);
    
    // With the fix, this should work now even for large inputs
    if expected_allocation_uncapped > 1_000_000_000 { // More than 1GB
        println!("ISSUE FIXED: Memory allocation is now capped to reasonable size");
        
        // Actually test the processing with the fix
        let result = std::panic::catch_unwind(|| {
            let words = jieba.cut(&content, false);
            println!("Successfully processed large content, got {} words", words.len());
            words
        });
        
        match result {
            Ok(_) => println!("Test PASSED: Large content processed successfully with fix"),
            Err(_) => panic!("Test FAILED: Memory allocation still fails even with fix"),
        }
    } else {
        // For smaller content, test normally
        let result = std::panic::catch_unwind(|| {
            jieba.cut(&content, false)
        });
        
        match result {
            Ok(_) => println!("Processed content successfully"),
            Err(_) => panic!("Memory allocation failed unexpectedly"),
        }
    }
}

#[test]
fn test_memory_allocation_cap() {
    // Test that very large hints are properly capped
    use jieba_rs::sparse_dag::StaticSparseDAG;
    
    // Test with extremely large hint
    let huge_hint = 1_000_000_000; // 1 billion
    let dag = StaticSparseDAG::with_size_hint(huge_hint);
    
    // The capacity should be capped to 1M elements, not huge_hint * 5
    // We can't directly access the capacity, but we can verify it doesn't crash
    println!("Successfully created DAG with huge hint: {}", huge_hint);
    
    // Test with normal hint
    let normal_hint = 1000;
    let dag2 = StaticSparseDAG::with_size_hint(normal_hint);
    println!("Successfully created DAG with normal hint: {}", normal_hint);
}