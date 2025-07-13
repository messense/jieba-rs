use jieba_rs::Jieba;

#[test]
fn test_moderate_size_with_fix() {
    // Test with a moderately large input that would have caused issues before the fix
    let jieba = Jieba::new();
    
    // Create a 10MB string (much more manageable than 250MB for testing)
    let base_text = "这是一个测试文本用来验证修复后的内存分配策略是否正常工作。";
    let content = base_text.repeat(200_000); // About 10MB
    
    println!("Content size: {} bytes", content.len());
    
    // Calculate expected allocations
    let heuristic_capacity = content.len() / 2;
    let uncapped_allocation = heuristic_capacity * 5 * std::mem::size_of::<usize>();
    let capped_allocation = std::cmp::min(heuristic_capacity * 5, 1_000_000) * std::mem::size_of::<usize>();
    
    println!("Heuristic capacity: {}", heuristic_capacity);
    println!("Uncapped allocation would be: {} bytes ({:.2} MB)", 
             uncapped_allocation, uncapped_allocation as f64 / 1_000_000.0);
    println!("Capped allocation: {} bytes ({:.2} MB)", 
             capped_allocation, capped_allocation as f64 / 1_000_000.0);
    
    // This should work fine with the fix
    let start = std::time::Instant::now();
    let words = jieba.cut(&content, false);
    let duration = start.elapsed();
    
    println!("Successfully processed {} bytes in {:?}, got {} words", 
             content.len(), duration, words.len());
    
    // Verify that we got reasonable output
    assert!(words.len() > 0);
    assert!(words.len() < content.len()); // Should have fewer words than characters
}