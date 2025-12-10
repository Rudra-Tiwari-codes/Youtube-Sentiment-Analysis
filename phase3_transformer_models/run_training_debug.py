"""Debug wrapper for training script"""
import sys
import traceback

print("="*60)
print("DEBUG: Starting training script...")
print("="*60)

try:
    # Import and run main
    print("DEBUG: Importing module...")
    exec(open('04_distilbert_training_real.py').read())
    print("\nDEBUG: Script completed successfully!")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
