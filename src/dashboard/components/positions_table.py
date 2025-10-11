"""
Positions Table Component for Trading Dashboard

This module provides components for displaying and managing trading positions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...core.data_models import Position, Trade, OrderSide


class PositionsTable:
    """Component for displaying and managing trading positions."""
    
    def __init__(self):
        self.color_scheme = {
            'profit': '#00ff88',
            'loss': '#ff4444',
            'neutral': '#888888',
            'buy': '#00ff88',
            'sell': '#ff4444'
        }
    
    def render_positions_table(self, 
                             positions: List[Position], 
                             show_actions: bool = True,
                             title: str = "Open Positions"):
        """
        Render positions table with optional action buttons.
        
        Args:
            positions: List of Position objects
            show_actions: Whether to show action buttons
            title: Table title
        """
        if not positions:
            st.info("No open positions")
            return
        
        st.subheader(title)
        
        # Convert positions to DataFrame
        df = self._positions_to_dataframe(positions)
        
        # Display metrics
        self._render_position_metrics(df)
        
        # Display table
        if show_actions:
            self._render_positions_table_with_actions(df)
        else:
            st.dataframe(df, use_container_width=True)
        
        # Display position charts
        self._render_position_charts(positions)
    
    def render_position_history(self, 
                              positions: List[Position], 
                              days: int = 30):
        """
        Render position history table.
        
        Args:
            positions: List of Position objects (including closed ones)
            days: Number of days to show
        """
        st.subheader("Position History")
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_positions = [p for p in positions if p.timestamp >= cutoff_date]
        
        if not recent_positions:
            st.info(f"No positions in the last {days} days")
            return
        
        # Convert to DataFrame
        df = self._positions_to_dataframe(recent_positions, include_closed=True)
        
        # Sort by timestamp (newest first)
        df = df.sort_values('Timestamp', ascending=False)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Display summary statistics
        self._render_position_history_summary(df)
    
    def render_position_details(self, position: Position):
        """
        Render detailed view of a single position.
        
        Args:
            position: Position object to display
        """
        st.subheader(f"Position Details - {position.symbol}")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Symbol", position.symbol)
        with col2:
            st.metric("Side", position.side.value)
        with col3:
            st.metric("Quantity", f"{position.quantity:.6f}")
        with col4:
            st.metric("Entry Price", f"${position.entry_price:.2f}")
        
        # PnL and metrics
        if position.current_price:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${position.current_price:.2f}")
            with col2:
                unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                st.metric("Unrealized PnL", f"${unrealized_pnl:.2f}")
            with col3:
                pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
                st.metric("PnL %", f"{pnl_pct:.2f}%")
            with col4:
                st.metric("Status", position.status.value)
        
        # Position timeline
        self._render_position_timeline(position)
    
    def _positions_to_dataframe(self, 
                               positions: List[Position], 
                               include_closed: bool = False) -> pd.DataFrame:
        """
        Convert positions to DataFrame.
        
        Args:
            positions: List of Position objects
            include_closed: Whether to include closed positions
        
        Returns:
            DataFrame with position data
        """
        data = []
        
        for pos in positions:
            if not include_closed and pos.status.value == 'closed':
                continue
            
            # Calculate PnL
            unrealized_pnl = 0
            pnl_pct = 0
            
            if pos.current_price:
                unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
            
            # Calculate time in position
            time_in_position = datetime.now() - pos.timestamp
            
            data.append({
                'Symbol': pos.symbol,
                'Side': pos.side.value,
                'Quantity': f"{pos.quantity:.6f}",
                'Entry Price': f"${pos.entry_price:.2f}",
                'Current Price': f"${pos.current_price:.2f}" if pos.current_price else "N/A",
                'Unrealized PnL': f"${unrealized_pnl:.2f}",
                'PnL %': f"{pnl_pct:.2f}%",
                'Status': pos.status.value,
                'Timestamp': pos.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Time in Position': str(time_in_position).split('.')[0]  # Remove microseconds
            })
        
        return pd.DataFrame(data)
    
    def _render_position_metrics(self, df: pd.DataFrame):
        """Render position summary metrics."""
        if df.empty:
            return
        
        # Calculate metrics
        total_positions = len(df)
        profitable_positions = len(df[df['Unrealized PnL'].str.contains('-', na=False) == False])
        total_unrealized_pnl = sum([float(pnl.replace('$', '').replace(',', '')) 
                                   for pnl in df['Unrealized PnL']])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Positions", total_positions)
        with col2:
            st.metric("Profitable", profitable_positions)
        with col3:
            win_rate = (profitable_positions / total_positions * 100) if total_positions > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Total Unrealized PnL", f"${total_unrealized_pnl:.2f}")
    
    def _render_positions_table_with_actions(self, df: pd.DataFrame):
        """Render positions table with action buttons."""
        # Create a copy for display
        display_df = df.copy()
        
        # Add action column
        display_df['Actions'] = ""
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
        # Add action buttons for each position
        for idx, row in df.iterrows():
            col1, col2, col3, col4 = st.columns([1, 1, 1, 7])
            
            with col1:
                if st.button(f"Close", key=f"close_{idx}"):
                    self._close_position(row['Symbol'])
            
            with col2:
                if st.button(f"Details", key=f"details_{idx}"):
                    st.session_state[f"show_details_{idx}"] = True
            
            with col3:
                if st.button(f"Modify", key=f"modify_{idx}"):
                    self._modify_position(row['Symbol'])
            
            with col4:
                # Show details if requested
                if st.session_state.get(f"show_details_{idx}", False):
                    self._render_position_details_modal(row)
    
    def _render_position_charts(self, positions: List[Position]):
        """Render position-related charts."""
        if not positions:
            return
        
        st.subheader("Position Analysis")
        
        # PnL distribution
        pnls = []
        for pos in positions:
            if pos.current_price:
                pnl = (pos.current_price - pos.entry_price) * pos.quantity
                pnls.append(pnl)
        
        if pnls:
            col1, col2 = st.columns(2)
            
            with col1:
                self._render_pnl_distribution(pnls)
            
            with col2:
                self._render_position_sizes_chart(positions)
    
    def _render_pnl_distribution(self, pnls: List[float]):
        """Render PnL distribution chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnls,
            nbinsx=20,
            name='PnL Distribution',
            marker_color=self.color_scheme['neutral'],
            opacity=0.7
        ))
        
        # Add mean line
        mean_pnl = np.mean(pnls)
        fig.add_vline(
            x=mean_pnl,
            line_dash="dash",
            line_color=self.color_scheme['profit'],
            annotation_text=f"Mean: ${mean_pnl:.2f}"
        )
        
        fig.update_layout(
            title="PnL Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_position_sizes_chart(self, positions: List[Position]):
        """Render position sizes chart."""
        symbols = [pos.symbol for pos in positions]
        sizes = [pos.quantity for pos in positions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=sizes,
            name='Position Sizes',
            marker_color=self.color_scheme['neutral']
        ))
        
        fig.update_layout(
            title="Position Sizes by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Quantity",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_position_history_summary(self, df: pd.DataFrame):
        """Render position history summary."""
        st.subheader("Summary Statistics")
        
        # Calculate summary metrics
        total_trades = len(df)
        closed_positions = len(df[df['Status'] == 'closed'])
        avg_pnl = 0  # Would need to calculate from actual PnL data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Closed Positions", closed_positions)
        with col3:
            st.metric("Avg PnL", f"${avg_pnl:.2f}")
        with col4:
            st.metric("Success Rate", "N/A")  # Would need to calculate
    
    def _render_position_details_modal(self, position_data: Dict[str, Any]):
        """Render position details in a modal-like format."""
        with st.expander(f"Position Details - {position_data['Symbol']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Symbol:** {position_data['Symbol']}")
                st.write(f"**Side:** {position_data['Side']}")
                st.write(f"**Quantity:** {position_data['Quantity']}")
                st.write(f"**Entry Price:** {position_data['Entry Price']}")
            
            with col2:
                st.write(f"**Current Price:** {position_data['Current Price']}")
                st.write(f"**Unrealized PnL:** {position_data['Unrealized PnL']}")
                st.write(f"**PnL %:** {position_data['PnL %']}")
                st.write(f"**Status:** {position_data['Status']}")
    
    def _render_position_timeline(self, position: Position):
        """Render position timeline."""
        st.subheader("Position Timeline")
        
        # This would normally show order history, modifications, etc.
        timeline_data = [
            {"Time": position.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
             "Event": "Position Opened", 
             "Details": f"Entry at ${position.entry_price:.2f}"}
        ]
        
        if position.current_price:
            timeline_data.append({
                "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Event": "Current Status",
                "Details": f"Current price: ${position.current_price:.2f}"
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
    
    def _close_position(self, symbol: str):
        """Close a position."""
        st.success(f"Closing position for {symbol}")
        # This would normally call the trading bot to close the position
    
    def _modify_position(self, symbol: str):
        """Modify a position."""
        st.info(f"Modifying position for {symbol}")
        # This would normally open a modification dialog
    
    def render_position_risk_metrics(self, positions: List[Position]):
        """Render position risk metrics."""
        if not positions:
            st.info("No positions for risk analysis")
            return
        
        st.subheader("Position Risk Analysis")
        
        # Calculate risk metrics
        total_exposure = sum([pos.quantity * pos.entry_price for pos in positions])
        max_position_size = max([pos.quantity * pos.entry_price for pos in positions]) if positions else 0
        position_concentration = (max_position_size / total_exposure * 100) if total_exposure > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Exposure", f"${total_exposure:.2f}")
        with col2:
            st.metric("Max Position Size", f"${max_position_size:.2f}")
        with col3:
            st.metric("Concentration", f"{position_concentration:.1f}%")
        with col4:
            st.metric("Position Count", len(positions))
        
        # Risk warnings
        if position_concentration > 50:
            st.warning("⚠️ High position concentration detected!")
        
        if len(positions) > 10:
            st.info("ℹ️ Consider reducing position count for better risk management")
    
    def render_position_performance(self, positions: List[Position]):
        """Render position performance analysis."""
        if not positions:
            st.info("No positions for performance analysis")
            return
        
        st.subheader("Position Performance")
        
        # Calculate performance metrics
        profitable_positions = [pos for pos in positions 
                              if pos.current_price and pos.current_price > pos.entry_price]
        losing_positions = [pos for pos in positions 
                           if pos.current_price and pos.current_price < pos.entry_price]
        
        win_rate = len(profitable_positions) / len(positions) * 100 if positions else 0
        
        # Average win/loss
        avg_win = np.mean([(pos.current_price - pos.entry_price) * pos.quantity 
                          for pos in profitable_positions]) if profitable_positions else 0
        avg_loss = np.mean([(pos.current_price - pos.entry_price) * pos.quantity 
                           for pos in losing_positions]) if losing_positions else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col2:
            st.metric("Avg Win", f"${avg_win:.2f}")
        with col3:
            st.metric("Avg Loss", f"${avg_loss:.2f}")
        with col4:
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        # Performance by symbol
        symbol_performance = {}
        for pos in positions:
            if pos.current_price:
                pnl = (pos.current_price - pos.entry_price) * pos.quantity
                if pos.symbol not in symbol_performance:
                    symbol_performance[pos.symbol] = []
                symbol_performance[pos.symbol].append(pnl)
        
        if symbol_performance:
            st.subheader("Performance by Symbol")
            
            symbol_data = []
            for symbol, pnls in symbol_performance.items():
                symbol_data.append({
                    'Symbol': symbol,
                    'Total PnL': f"${sum(pnls):.2f}",
                    'Avg PnL': f"${np.mean(pnls):.2f}",
                    'Positions': len(pnls)
                })
            
            symbol_df = pd.DataFrame(symbol_data)
            st.dataframe(symbol_df, use_container_width=True)
