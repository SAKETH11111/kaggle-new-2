#!/bin/bash

echo "🚀 FlightRank 2025 Training Monitor"
echo "=================================="
echo "Training started at: $(date)"
echo ""

# Function to show progress
show_progress() {
    echo "📊 Current Status:"
    if ps aux | grep -q "python3 train_and_validate.py" | grep -v grep; then
        echo "✅ Training process is RUNNING"
        
        # Show resource usage
        echo "🔥 Resource Usage:"
        ps aux | grep "python3 train_and_validate.py" | grep -v grep | awk '{print "   CPU: " $3 "%, Memory: " $4 "%, PID: " $2}'
        
        # Show log tail
        echo ""
        echo "📝 Latest Training Output:"
        echo "-------------------------"
        tail -15 training_output.log
        
    else
        echo "⭕ Training process has FINISHED"
        echo ""
        echo "📋 Final Results:"
        echo "=================="
        cat training_output.log | grep -E "(HitRate@3|Average|SUCCESS|WARNING|ERROR|Fold [0-9]/5|✅|❌|🎯)"
        
        # Check if model was saved
        if [ -f "./models/best_model.pkl" ]; then
            echo ""
            echo "✅ Model saved successfully: $(ls -lh ./models/best_model.pkl)"
        else
            echo ""
            echo "❌ Model file not found"
        fi
    fi
}

# Monitor continuously
while true; do
    clear
    show_progress
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    sleep 10
done