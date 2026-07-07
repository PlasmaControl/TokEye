#!/usr/bin/env python3
"""
Command Line Interface for eigspec - Python port of eigspec_mmain.m

This module provides an interactive command-line interface for spectral analysis
and modal identification, replicating the functionality of the MATLAB eigspec_mmain.m.

Features:
- Interactive command prompt with help system
- Script execution capability  
- Data loading and preprocessing
- Spectral analysis workflows
- Assessment and clustering tools
- Visualization and export functions

Commands mirror the MATLAB version where possible for familiarity.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

# eigspec imports
from . import __version__
from .utils.signal_processing import ar_assessment, correlation_assessment, get_performance_info
from .utils.modal_analysis import complex_vector_scalar_fit, modal_mac_matrix
from .analysis.random_projection import random_projection_spectral_analysis


class EigspecCLI:
    """Main command-line interface for eigspec analysis."""
    
    def __init__(self) -> None:
        """Initialize the CLI with default settings."""
        self.prompt = "-->"
        self.data: Dict[str, Any] = {}
        self.settings: Dict[str, Any] = {
            # Default analysis parameters (equivalent to MATLAB defaults)
            'fft_options': {
                'bss': [512, 128],      # [block_size, block_stride]
                'nfft': 2048,
                'nsmooth': 5,
                'rpdim': 0,
                'winstr': 'hamming'
            },
            'rndspec_options': {
                'bss': [500, 400],
                'rfpn': [-12, 10, 20, 12, 20],  # random projection parameters
                'thresh': [0.998, 0.998]
            },
            'clus_options': {
                'numrestarts': 10,
                'kr': [40, 3],          # k-NN parameters
                'knn': 40
            },
            'MN_MMAX': 8,               # Maximum toroidal mode number
            'MN_NMAX': 5,               # Maximum poloidal mode number
            'default_ar_lag': 12,
            'default_filter_order': 2
        }
        
        # Analysis results storage
        self.current_data = None        # Raw data
        self.processed_data = None      # Preprocessed data 
        self.analysis_results = None    # Spectral analysis results
        self.assessment_results = None  # Channel assessment results
        self.cluster_results = None     # Clustering results
        
        # Command registry
        self.commands: Dict[str, callable] = {
            'help': self.cmd_help,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,
            'version': self.cmd_version,
            'status': self.cmd_status,
            'settings': self.cmd_settings,
            'load': self.cmd_load,
            'assess': self.cmd_assess,
            'corr': self.cmd_corr,
            'rndspec': self.cmd_rndspec,
            'view': self.cmd_view,
            'export': self.cmd_export,
            'script': self.cmd_script,
            'clear': self.cmd_clear
        }
        
        self.running = True
        self.script_mode = False
        self.script_commands: List[str] = []
        self.script_index = 0
    
    def run(self, script_file: Optional[str] = None) -> None:
        """Run the CLI interface.
        
        Args:
            script_file: Optional script file to execute on startup
        """
        self._print_banner()
        
        if script_file:
            self.cmd_script([script_file])
        
        while self.running:
            try:
                if self.script_mode:
                    if self.script_index < len(self.script_commands):
                        command_line = self.script_commands[self.script_index]
                        print(f"[@script/line #{self.script_index+1}]{self.prompt}{command_line}")
                        self.script_index += 1
                    else:
                        print("[script done.]")
                        self.script_mode = False
                        continue
                else:
                    command_line = input(f"{self.prompt}").strip()
                
                if command_line:
                    self._execute_command(command_line)
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
                continue
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                if os.getenv('EIGSPEC_DEBUG'):
                    traceback.print_exc()
    
    def _print_banner(self) -> None:
        """Print startup banner."""
        import platform
        
        print("<<< entering eigspec: (array magnetics fluctuation analysis)")
        print(f"<<< version: {__version__} (Python)")
        print(f"<<< platform: {platform.system()} {platform.release()}")
        print(f"<<< working directory: {os.getcwd()}")
        print("<<< type 'help' for information, or 'quit' to exit")
        
        # Show performance info
        perf_info = get_performance_info()
        if perf_info['numba_available']:
            print("<<< performance: Numba JIT compilation available")
        if perf_info['joblib_available']:
            print("<<< performance: parallel processing available")
    
    def _execute_command(self, command_line: str) -> None:
        """Execute a command line."""
        # Skip comments
        if command_line.startswith('//'):
            return
        
        # Parse command and arguments
        parts = command_line.split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in self.commands:
            try:
                self.commands[command](args)
            except Exception as e:
                print(f"Error executing '{command}': {e}")
                if os.getenv('EIGSPEC_DEBUG'):
                    traceback.print_exc()
        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")
    
    def cmd_help(self, args: List[str]) -> None:
        """Show help information."""
        if not args:
            print("eigspec - Spectral analysis and modal identification")
            print("\nAvailable commands:")
            print("  Data I/O:")
            print("    load <file>           - Load data from file")
            print("    export <format> <file> - Export results")
            print("    clear                 - Clear loaded data")
            print("")
            print("  Analysis:")
            print("    assess <lag> [bsize] [bstride] - AR-based channel assessment")
            print("    corr [bsize] [bstride]         - Correlation assessment")
            print("    rndspec [options]              - Random projection spectral analysis")
            print("")
            print("  Utilities:")
            print("    view [results]        - View analysis results")
            print("    settings [param] [value] - Show/set analysis parameters")
            print("    status               - Show current status")
            print("    script <file>        - Execute script file")
            print("")
            print("  System:")
            print("    version              - Show version information")
            print("    help [command]       - Show help")
            print("    quit                 - Exit program")
        else:
            command = args[0].lower()
            if command == 'load':
                print("load <file> - Load data from file")
                print("  Supported formats: .npy, .npz, .csv, .txt")
                print("  Data should be organized as (n_samples, n_channels)")
            elif command == 'assess':
                print("assess <ar_lag> [block_size] [block_stride] - AR-based assessment")
                print("  ar_lag: AR model lag order (e.g., 12)")
                print("  block_size: Analysis block size (default: 800)")
                print("  block_stride: Block stride (default: 800)")
            elif command == 'rndspec':
                print("rndspec [reduced_dim] [future] [past] [order1] [order2] - Spectral analysis")
                print("  All parameters optional, uses settings defaults if not provided")
                print("  reduced_dim: Random projection dimension (<0=orthonormal, >0=random)")
                print("  future/past: SSI horizon parameters")
                print("  order1/order2: Model orders")
            else:
                print(f"No detailed help available for '{command}'")
    
    def cmd_quit(self, args: List[str]) -> None:
        """Exit the program."""
        print("Goodbye!")
        self.running = False
    
    def cmd_version(self, args: List[str]) -> None:
        """Show version information."""
        import platform
        
        print(f"eigspec version: {__version__}")
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.system()} {platform.release()}")
        
        perf_info = get_performance_info()
        print(f"Numba available: {perf_info['numba_available']}")
        print(f"Joblib available: {perf_info['joblib_available']}")
    
    def cmd_status(self, args: List[str]) -> None:
        """Show current analysis status."""
        print("=== Analysis Status ===")
        
        if self.current_data is not None:
            n_samples, n_channels = self.current_data.shape
            print(f"Data loaded: {n_samples} samples, {n_channels} channels")
        else:
            print("Data: None loaded")
        
        if self.analysis_results is not None:
            n_blocks = len(self.analysis_results.block_results)
            print(f"Spectral analysis: {n_blocks} blocks processed")
        else:
            print("Spectral analysis: Not performed")
        
        if self.assessment_results is not None:
            print("Assessment: Available")
        else:
            print("Assessment: Not performed")
    
    def cmd_settings(self, args: List[str]) -> None:
        """Show or modify analysis settings."""
        if not args:
            print("=== Current Settings ===")
            for category, params in self.settings.items():
                print(f"{category}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {params}")
        elif len(args) == 1:
            # Show specific setting
            param = args[0]
            found = False
            for category, params in self.settings.items():
                if isinstance(params, dict) and param in params:
                    print(f"{category}.{param}: {params[param]}")
                    found = True
                elif param == category:
                    print(f"{category}: {params}")
                    found = True
            if not found:
                print(f"Setting '{param}' not found")
        else:
            print("Usage: settings [parameter] [value]")
    
    def cmd_load(self, args: List[str]) -> None:
        """Load data from file."""
        if not args:
            print("Usage: load <filename>")
            return
        
        filename = args[0]
        
        try:
            if filename.endswith('.npy'):
                data = np.load(filename)
            elif filename.endswith('.npz'):
                archive = np.load(filename)
                # Try common key names
                for key in ['data', 'signals', 'y', 'Y']:
                    if key in archive:
                        data = archive[key]
                        break
                else:
                    # Use first array found
                    data = next(iter(archive.values()))
            elif filename.endswith(('.csv', '.txt')):
                data = np.loadtxt(filename, delimiter=',')
            else:
                print(f"Unsupported file format: {filename}")
                return
            
            # Ensure 2D array
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            self.current_data = data
            n_samples, n_channels = data.shape
            print(f"Loaded {filename}: {n_samples} samples, {n_channels} channels")
            
            # Clear previous analysis results
            self.analysis_results = None
            self.assessment_results = None
            self.cluster_results = None
            
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    
    def cmd_assess(self, args: List[str]) -> None:
        """Perform AR-based channel assessment."""
        if self.current_data is None:
            print("No data loaded. Use 'load <file>' first.")
            return
        
        if not args:
            # Use default parameters
            ar_lag = self.settings['default_ar_lag']
            block_size = self.settings['rndspec_options']['bss'][0]
            block_stride = self.settings['rndspec_options']['bss'][1]
        else:
            try:
                ar_lag = int(args[0])
                block_size = int(args[1]) if len(args) > 1 else self.settings['rndspec_options']['bss'][0]
                block_stride = int(args[2]) if len(args) > 2 else self.settings['rndspec_options']['bss'][1]
            except ValueError:
                print("Usage: assess <ar_lag> [block_size] [block_stride]")
                return
        
        print(f"AR-based assessment: lag={ar_lag}, block=[{block_size}, {block_stride}]")
        
        try:
            self.assessment_results = ar_assessment(
                time_vector=None,
                signal_array=self.current_data,
                ar_lag=ar_lag,
                block_parameters=(block_size, block_stride)
            )
            
            print("Assessment complete. Results:")
            print(f"  Median predictability: {self.assessment_results.median_scores[:, 0].mean():.3f}")
            print(f"  Median participation: {self.assessment_results.median_scores[:, 1].mean():.3f}")
            print("Use 'view assess' to see detailed results.")
            
        except Exception as e:
            print(f"Assessment failed: {e}")
    
    def cmd_corr(self, args: List[str]) -> None:
        """Perform correlation-based assessment."""
        if self.current_data is None:
            print("No data loaded. Use 'load <file>' first.")
            return
        
        if args:
            try:
                block_size = int(args[0])
                block_stride = int(args[1]) if len(args) > 1 else block_size
            except ValueError:
                print("Usage: corr [block_size] [block_stride]")
                return
        else:
            block_size = self.settings['rndspec_options']['bss'][0]
            block_stride = self.settings['rndspec_options']['bss'][1]
        
        print(f"Correlation assessment: block=[{block_size}, {block_stride}]")
        
        try:
            corr_result = correlation_assessment(
                time_vector=None,
                signal_array=self.current_data,
                block_parameters=(block_size, block_stride)
            )
            
            # Store result
            self.data['correlation_result'] = corr_result
            
            # Show summary
            corr_matrix = corr_result.correlation_matrix
            off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            print("Correlation assessment complete:")
            print(f"  Mean correlation: {off_diag.mean():.3f}")
            print(f"  Max correlation: {off_diag.max():.3f}")
            print(f"  Min correlation: {off_diag.min():.3f}")
            
        except Exception as e:
            print(f"Correlation assessment failed: {e}")
    
    def cmd_rndspec(self, args: List[str]) -> None:
        """Perform random projection spectral analysis."""
        if self.current_data is None:
            print("No data loaded. Use 'load <file>' first.")
            return
        
        # Parse parameters or use defaults
        if args:
            try:
                reduced_dim = int(args[0])
                future = int(args[1]) if len(args) > 1 else self.settings['rndspec_options']['rfpn'][1]
                past = int(args[2]) if len(args) > 2 else self.settings['rndspec_options']['rfpn'][2]
                order1 = int(args[3]) if len(args) > 3 else self.settings['rndspec_options']['rfpn'][3]
                order2 = int(args[4]) if len(args) > 4 else self.settings['rndspec_options']['rfpn'][4]
            except ValueError:
                print("Usage: rndspec [reduced_dim] [future] [past] [order1] [order2]")
                return
        else:
            reduced_dim = self.settings['rndspec_options']['rfpn'][0]
            future = self.settings['rndspec_options']['rfpn'][1]
            past = self.settings['rndspec_options']['rfpn'][2]
            order1 = self.settings['rndspec_options']['rfpn'][3]
            order2 = self.settings['rndspec_options']['rfpn'][4]
        
        # Determine if SSI or AR/PCA based on parameters
        if len(args) >= 3 or future != past:
            # SSI mode
            reduced_dimension = [reduced_dim, future, past, order1, order2]
            use_cca = False
        else:
            # AR/PCA mode
            reduced_dimension = [reduced_dim, past, order1, order2]
            use_cca = False
        
        block_parameters = tuple(self.settings['rndspec_options']['bss'])
        threshold_parameters = tuple(self.settings['rndspec_options']['thresh'])
        
        print(f"Random projection analysis: reduced_dim={reduced_dimension}")
        print(f"Block parameters: {block_parameters}")
        
        try:
            self.analysis_results = random_projection_spectral_analysis(
                time_array=None,
                signal_array=self.current_data,
                block_parameters=block_parameters,
                reduced_dimension=reduced_dimension,
                threshold_parameters=threshold_parameters,
                use_canonical_correlation_analysis=use_cca
            )
            
            n_blocks = len(self.analysis_results.block_results)
            total_time = self.analysis_results.total_processing_time
            
            print(f"Analysis complete: {n_blocks} blocks processed in {total_time:.2f}s")
            print("Use 'view results' to examine the results.")
            
        except Exception as e:
            print(f"Spectral analysis failed: {e}")
    
    def cmd_view(self, args: List[str]) -> None:
        """View analysis results."""
        if not args:
            print("Usage: view <results|assess|corr>")
            return
        
        view_type = args[0].lower()
        
        if view_type == 'results':
            if self.analysis_results is None:
                print("No spectral analysis results available.")
                return
            self._view_spectral_results()
        
        elif view_type == 'assess':
            if self.assessment_results is None:
                print("No assessment results available.")
                return
            self._view_assessment_results()
        
        elif view_type == 'corr':
            if 'correlation_result' not in self.data:
                print("No correlation results available.")
                return
            self._view_correlation_results()
        
        else:
            print(f"Unknown view type: {view_type}")
    
    def _view_spectral_results(self) -> None:
        """Display spectral analysis results summary."""
        results = self.analysis_results
        print("=== Spectral Analysis Results ===")
        print(f"Analysis type: {results.analysis_name}")
        print(f"Total blocks: {len(results.block_results)}")
        print(f"Processing time: {results.total_processing_time:.2f}s")
        print(f"Block parameters: {results.block_parameters}")
        print(f"Reduced dimension: {results.reduced_dimension}")
        print(f"Thresholds: {results.threshold_parameters}")
        
        # Summary of modes found
        total_modes = 0
        for block_result in results.block_results:
            if hasattr(block_result.reduced_dimension_matrix, 'entries'):
                total_modes += len(block_result.reduced_dimension_matrix.entries)
        
        print(f"Total modes identified: {total_modes}")
    
    def _view_assessment_results(self) -> None:
        """Display assessment results summary."""
        result = self.assessment_results
        print("=== AR Assessment Results ===")
        print(f"Channels: {len(result.channel_names)}")
        print(f"AR lag: {result.ar_lag}")
        print(f"Block parameters: {result.block_parameters}")
        
        print("\nChannel Summary (median values):")
        print("Ch#  Predictability  Participation  RMS")
        print("-" * 45)
        
        for i, (pred, part, rms) in enumerate(result.median_scores):
            ch_name = result.channel_names[i].split('(')[0].strip()
            print(f"{i+1:3d}  {pred:12.3f}  {part:12.3f}  {rms:8.3e}")
    
    def _view_correlation_results(self) -> None:
        """Display correlation results summary."""
        result = self.data['correlation_result']
        corr_matrix = result.correlation_matrix
        
        print("=== Correlation Assessment Results ===")
        print(f"Channels: {len(result.channel_names)}")
        
        # Show correlation matrix
        n_channels = len(result.channel_names)
        if n_channels <= 10:
            print("\nCorrelation Matrix:")
            print("   ", end="")
            for i in range(n_channels):
                print(f"{i+1:6d}", end="")
            print()
            
            for i in range(n_channels):
                print(f"{i+1:3d}", end="")
                for j in range(n_channels):
                    print(f"{corr_matrix[i,j]:6.2f}", end="")
                print()
        else:
            print("(Correlation matrix too large to display)")
    
    def cmd_export(self, args: List[str]) -> None:
        """Export analysis results."""
        if len(args) < 2:
            print("Usage: export <format> <filename>")
            print("Formats: csv, npy, npz")
            return
        
        format_type = args[0].lower()
        filename = args[1]
        
        try:
            if format_type == 'csv':
                if self.assessment_results is not None:
                    # Export assessment results as CSV
                    data = np.column_stack([
                        np.arange(1, len(self.assessment_results.channel_names) + 1),
                        self.assessment_results.median_scores
                    ])
                    header = "Channel,Predictability,Participation,RMS"
                    np.savetxt(filename, data, delimiter=',', header=header, fmt='%g')
                    print(f"Assessment results exported to {filename}")
                else:
                    print("No assessment results to export")
            
            elif format_type in ['npy', 'npz']:
                # Export all available data
                export_data = {}
                
                if self.current_data is not None:
                    export_data['raw_data'] = self.current_data
                
                if self.assessment_results is not None:
                    export_data['assessment_median'] = self.assessment_results.median_scores
                    export_data['assessment_full'] = {
                        'predictability': self.assessment_results.predictability,
                        'participation': self.assessment_results.participation,
                        'rms': self.assessment_results.rms_values
                    }
                
                if format_type == 'npy' and len(export_data) == 1:
                    np.save(filename, next(iter(export_data.values())))
                else:
                    np.savez(filename, **export_data)
                
                print(f"Data exported to {filename}")
            
            else:
                print(f"Unsupported format: {format_type}")
        
        except Exception as e:
            print(f"Export failed: {e}")
    
    def cmd_script(self, args: List[str]) -> None:
        """Execute commands from script file."""
        if not args:
            print("Usage: script <filename>")
            return
        
        script_file = args[0]
        
        try:
            with open(script_file, 'r') as f:
                commands = [line.strip() for line in f if line.strip() and not line.startswith('//')]
            
            self.script_commands = commands
            self.script_index = 0
            self.script_mode = True
            
            print(f"Loaded script: {script_file} ({len(commands)} commands)")
        
        except Exception as e:
            print(f"Failed to load script {script_file}: {e}")
    
    def cmd_clear(self, args: List[str]) -> None:
        """Clear loaded data and results."""
        self.current_data = None
        self.analysis_results = None
        self.assessment_results = None
        self.cluster_results = None
        self.data.clear()
        print("All data and results cleared.")


def main() -> None:
    """Main entry point for eigspec CLI."""
    parser = argparse.ArgumentParser(
        description="eigspec - Spectral analysis and modal identification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'script', 
        nargs='?', 
        help='Optional script file to execute'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'eigspec {__version__}'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed error messages'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ['EIGSPEC_DEBUG'] = '1'
    
    cli = EigspecCLI()
    cli.run(script_file=args.script)


if __name__ == "__main__":
    main() 