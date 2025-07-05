import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir):
    """Load TensorBoard event files into a pandas DataFrame"""
    event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    # Create a DataFrame to store all metrics
    metrics_df = pd.DataFrame()
    
    for event_file in event_files:
        event_acc = EventAccumulator(str(event_file))
        event_acc.Reload()
        
        tags = event_acc.Tags()['scalars']
        data = {}
        
        for tag in tags:
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = values
            data['step'] = steps
        
        # Create DataFrame for this event file
        file_df = pd.DataFrame(data)
        file_df['file'] = event_file.name
        metrics_df = pd.concat([metrics_df, file_df], ignore_index=True)
    
    return metrics_df

def plot_training_history(df, output_dir):
    """Create loss and metrics evolution plots"""
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(df['step'], df['loss'], label='Training Loss')
    if 'eval_loss' in df.columns:
        plt.plot(df['step'], df['eval_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    if 'eval_accuracy' in df.columns:
        plt.subplot(2, 2, 2)
        plt.plot(df['step'], df['eval_accuracy'])
        plt.title('Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.grid(True)
    
    # F1 score plot
    if 'eval_f1' in df.columns:
        plt.subplot(2, 2, 3)
        plt.plot(df['step'], df['eval_f1'])
        plt.title('Validation F1 Score')
        plt.xlabel('Step')
        plt.ylabel('F1 Score')
        plt.grid(True)
    
    # Learning rate plot
    if 'learning_rate' in df.columns:
        plt.subplot(2, 2, 4)
        plt.plot(df['step'], df['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved training history plot to {plot_path}')

def plot_metric_correlations(df, output_dir):
    """Analyze correlations between different metrics"""
    # Select relevant columns
    corr_columns = [col for col in df.columns if col not in ['step', 'file']]
    
    if len(corr_columns) < 2:
        print("Not enough metrics for correlation analysis")
        return
    
    # Compute correlation matrix
    corr_matrix = df[corr_columns].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Metric Correlations')
    
    plot_path = os.path.join(output_dir, 'metric_correlations.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved metric correlations plot to {plot_path}')

def plot_loss_landscape(df, output_dir):
    """Visualize the training loss landscape"""
    if 'loss' not in df.columns:
        print("No loss data available")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(df['step'], df['loss'])
    plt.title('Training Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Loss distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['loss'], kde=True)
    plt.title('Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_landscape.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved loss landscape plot to {plot_path}')

def plot_lr_analysis(df, output_dir):
    """Analyze learning rate impact on training"""
    if 'learning_rate' not in df.columns or 'loss' not in df.columns:
        print("Missing data for LR analysis")
        return
    
    plt.figure(figsize=(12, 10))
    
    # LR vs Loss
    plt.subplot(2, 2, 1)
    plt.scatter(df['learning_rate'], df['loss'], alpha=0.5)
    plt.title('Learning Rate vs Training Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # LR schedule
    plt.subplot(2, 2, 2)
    plt.plot(df['step'], df['learning_rate'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # Log-scale LR vs Loss
    plt.subplot(2, 2, 3)
    plt.scatter(np.log10(df['learning_rate']), df['loss'], alpha=0.5)
    plt.title('Log Learning Rate vs Loss')
    plt.xlabel('log10(Learning Rate)')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Smoothed loss with LR
    plt.subplot(2, 2, 4)
    plt.plot(df['step'], df['loss'], label='Loss')
    plt.twinx()
    plt.plot(df['step'], df['learning_rate'], 'r-', label='Learning Rate')
    plt.title('Loss and Learning Rate Evolution')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'lr_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved learning rate analysis plot to {plot_path}')

def plot_evaluation_metrics(df, output_dir):
    """Visualize evaluation metrics over time"""
    eval_metrics = [col for col in df.columns if col.startswith('eval_')]
    
    if not eval_metrics:
        print("No evaluation metrics found")
        return
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(eval_metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(df['step'], df[metric])
        plt.title(f'{metric.replace("eval_", "").title()} Evolution')
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.grid(True)
        
        if i >= 4:  # Only show 4 metrics at a time
            break
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved evaluation metrics plot to {plot_path}')

def main(log_dir, output_dir='vis'):
    """Main function to generate all visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load TensorBoard data
        df = load_tensorboard_data(log_dir)
        
        if df.empty:
            print("No data loaded from TensorBoard event files")
            return
            
        # Print available metrics
        print("Available metrics in TensorBoard data:")
        print([col for col in df.columns if col not in ['step', 'file']])
        
        # Generate visualizations
        plot_training_history(df, output_dir)
        plot_metric_correlations(df, output_dir)
        plot_loss_landscape(df, output_dir)
        plot_lr_analysis(df, output_dir)
        plot_evaluation_metrics(df, output_dir)
        
        print("\nVisualization Report:")
        print(f"- TensorBoard logs analyzed from: {log_dir}")
        print(f"- Visualizations saved to: {output_dir}")
        
        return df
        
    except Exception as e:
        print(f"Error processing TensorBoard data: {e}")
        return None

if __name__ == "__main__":
    # Update with your actual log directory
    log_dir = "./checkpoints/distilroberta-base_subtaskB/logs"
    
    # Run the visualization pipeline
    df = main(log_dir)
    
    # Optional: Save the processed data
    if df is not None:
        df.to_csv(os.path.join('vis', 'tensorboard_metrics.csv'), index=False)
        print("Saved processed metrics data to vis/tensorboard_metrics.csv")