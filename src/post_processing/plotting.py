import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
import cv2
from mergeddata import MergedData
from validation import Validation as Val

class Plotting:
    @staticmethod
    def _get_lim():
        return -10, 30

    @staticmethod
    def plot_line(series: pd.Series,
                  xlabel: str,
                  ylabel: str,
                  title: str,
                  figsize: tuple[int] = (12,6)):
        Val.validate_type(series, pd.Series, "Series")
        Val.validate_strings(xlabel=xlabel, ylabel=ylabel, title=title)

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(series, marker='o', linestyle='-', color='b')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.show()
        except Exception as e:
            raise Exception(f"Error plotting line: {e}")

    @staticmethod
    def plot_lines(df: pd.DataFrame,
                   columns: list[str],
                   xlabel: str,
                   ylabel_1: str,
                   ylabel_2: str,
                   title: str,
                   figsize: tuple[int] = (12,6)):
        Val.validate_dataframe(df, required_columns=columns, name="DataFrame")
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_strings(xlabel=xlabel, ylabel_1=ylabel_1,
                             ylabel_2=ylabel_2, title=title)

        try:
            # plot two axes on the same plot
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = ax1.twinx()

            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel_1, color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.set_ylabel(ylabel_2, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax1.set_title(title)

            ax1.plot(df[columns[0]], linestyle='-', color='tab:blue')
            ax2.plot(df[columns[1]], linestyle='-', color='tab:red')

            fig.tight_layout()
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

            plt.title(title)
            plt.show()
        except Exception as e:
            raise Exception(f"Error plotting lines: {e}")

    @staticmethod
    def plot_homography_animated(homography_points: np.ndarray,
                                 df_transformed_monofil: pd.DataFrame,
                                 filepath: str,
                                 fps: int=30,
                                 figsize: tuple[int] = (12,12)):
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_type(df_transformed_monofil, pd.DataFrame, "DataFrame")
        Val.validate_path(filepath, file_types=[".mp4", ".avi"])
        Val.validate_type(fps, int, "FPS")
        Val.validate_positive(fps, "FPS")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlim(Plotting._get_lim())
            ax.set_ylim(Plotting._get_lim())
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')

            # Plot destination points
            for point in homography_points:
                ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

            line, = ax.plot([], [], 'bo-')

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                points = df_transformed_monofil.iloc[frame].values.reshape(-1, 2)
                line.set_data(points[:, 0], points[:, 1])  # Convert to mm
                return line,

            anim = FuncAnimation(fig, update, frames=len(df_transformed_monofil), init_func=init, blit=True, interval=1000/fps)
            plt.show()
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            raise Exception(f"Error creating animation: {e}")


    # For synchronized data
    @staticmethod
    def plot_rf_mapping_animated(merged_data: MergedData,
                                x_col: str, y_col: str,
                                homography_points: np.ndarray,
                                size_col: str,
                                color_col: str,
                                filepath: str,
                                bending: bool = False,
                                spikes: bool = False,
                                xlabel: str = "x (mm)",
                                ylabel: str = "y (mm)",
                                fps: int = 30,
                                figsize: tuple[int] = (12, 12),
                                cmap: str = "viridis"):
        Val.validate_type(merged_data, MergedData, "MergedData")
        Val.validate_strings(x_col=x_col, y_col=y_col,
                             size_col=size_col, colour_col=color_col,
                             xlabel=xlabel, ylabel=ylabel, cmap=cmap)
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_path(filepath, file_types=[".mp4", ".avi"])
        Val.validate_type(bending, bool, "Bending")
        Val.validate_type(spikes, bool, "Spikes")
        Val.validate_type(fps, int, "FPS")
        Val.validate_positive(fps, "FPS")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            df = merged_data.threshold_data(bending, spikes)
            ax.set_xlim(Plotting._get_lim())
            ax.set_ylim(Plotting._get_lim())
            ax.set_title('RF Mapping Animation')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Plot reference grid (homography points)
            for point in homography_points:
                ax.axhline(y=point[1] , color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=point[0] , color='gray', linestyle='--', alpha=0.5)

            # Normalize size
            scaler = MinMaxScaler(feature_range=(5, 30))
            df["scaled_size"] = scaler.fit_transform(df[[size_col]])

            # Prepare color mapping
            if color_col == "Spikes":
                color_map = df["Spikes"].apply(lambda x: 'blue' if x > 0 else 'grey')
            else:
                color_norm = plt.Normalize(df[color_col].min(), df[color_col].max())
                color_mapper = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
                color_map = color_mapper.to_rgba(df[color_col])

            circles = []

            # Update function
            def update(frame):
                current_row = df.iloc[frame]
                x, y = current_row[x_col] , current_row[y_col] 
                size = current_row["scaled_size"] * 0.015
                color = color_map[frame] if color_col != "Spikes" else color_map.iloc[frame]

                circle = plt.Circle((x, y), size, color=color, alpha=0.5, edgecolor="k", linewidth=0.5)
                ax.add_patch(circle)
                circles.append(circle)
                return circle,

            anim = FuncAnimation(fig, update, frames=len(df), interval=1000 / fps, blit=False)

            if color_col == "Spikes":
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Spike'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='No Spike')
                ]
                ax.legend(handles=legend_elements,
                        loc="upper left",
                        title=f"Spike Status\n(Circle Size ∝ {size_col})")
            else:
                cbar = fig.colorbar(color_mapper, ax=ax)
                cbar.set_label(f'{color_col} (Color)')

                # Add a small text box for size explanation
                text_box = f"Circle Size ∝ {size_col}"
                ax.text(0.02, 0.98, text_box, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

            plt.show()
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])

        except Exception as e:
            raise Exception(f"Error creating animation: {e}")

    @staticmethod
    def background_framing(merged_data: MergedData,
                           ax: plt.Axes,
                           homography_points: np.ndarray,
                           video_path: str = None,
                           index: int = None):
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        if video_path is not None and index is not None:
            Val.validate_path(video_path, file_types=[".mp4", ".avi"])
            Val.validate_type(index, int, "Index")
            Val.validate_positive(index, "Index", zero_allowed=True)

            dst_min, dst_max = 300, 500
            dst_points = np.array([[dst_min, dst_max],
                                   [dst_max, dst_max],
                                   [dst_max, dst_min],
                                   [dst_min, dst_min]])
            h_matrix = merged_data.dlc._get_homography_matrix(index, dst_points)

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("Error: Could not read video frame.")
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            frame_transformed = cv2.warpPerspective(frame, h_matrix, (w, h))

            pixel_to_mm = 0.1
            frame_width_mm = w * pixel_to_mm
            frame_height_mm = h * pixel_to_mm
            offset_x_mm = -30
            offset_y_mm = -30

            extent = [offset_x_mm,
                      offset_x_mm + frame_width_mm,
                      offset_y_mm + frame_height_mm,
                      offset_y_mm]
            ax.imshow(frame_transformed, extent=extent)

        for point in homography_points:
            ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

    @staticmethod
    def plot_kde_density(merged_data: MergedData,
                         x_col: str, y_col: str,
                         homography_points: np.ndarray,
                         bending: bool = False,
                         spikes: bool = False,
                         title: str = 'KDE Plot',
                         xlabel: str = 'x (mm)', ylabel: str = 'y (mm)',
                         figsize: tuple[int] = (12, 12),
                         cmap = "vlag",
                         # Video frame options
                         frame: bool = False,
                         video_path: str = None,
                         index: int = None):

        Val.validate_type(merged_data, MergedData, "MergedData")
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_strings(x_col=x_col, y_col=y_col,
                             xlabel=xlabel, ylabel=ylabel, title=title,
                             cmap=cmap)
        Val.validate_type(bending, bool, "Bending")
        Val.validate_type(spikes, bool, "Spikes")

        fig, ax = plt.subplots(figsize=figsize)
        Plotting.background_framing(merged_data, ax, homography_points, video_path if frame else None, index if frame else None)

        df = merged_data.threshold_data(bending, spikes)

        sns.kdeplot(x=df[x_col], y=df[y_col],
                    fill=True, cmap=cmap, bw_adjust=0.3, ax=ax, alpha=0.5)

        ax.set_xlim(Plotting._get_lim())
        ax.set_ylim(Plotting._get_lim())
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.show()
        return fig, ax

    @staticmethod
    def plot_scatter(merged_data: MergedData,
                    x_col: str, y_col: str,
                    homography_points: np.ndarray,
                    size_col: str,
                    color_col: str = None,  # New parameter for color mapping
                    bending: bool = False,
                    spikes: bool = False,
                    title: str = 'Scatter Plot',
                    legend_title: str = 'Neuron Spike Status',
                    xlabel: str = 'x (mm)', ylabel: str = 'y (mm)',
                    figsize: tuple[int] = (12, 12),
                    cmap: str = 'viridis',
                    # Video frame options
                    frame: bool = False,
                    video_path: str = None,
                    index: int = None):

        Val.validate_type(merged_data, MergedData, "MergedData")
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_strings(x_col=x_col, y_col=y_col,
                             size_col=size_col, color_col=color_col,
                             xlabel=xlabel, ylabel=ylabel,
                             title=title, legend_title=legend_title,
                             cmap=cmap)
        Val.validate_type(bending, bool, "Bending")
        Val.validate_type(spikes, bool, "Spikes")

        fig, ax = plt.subplots(figsize=figsize)
        Plotting.background_framing(merged_data, ax, homography_points, video_path if frame else None, index if frame else None)

        df = merged_data.threshold_data(bending, spikes)

        # Normalize sizes
        norm = plt.Normalize(df[size_col].min(), df[size_col].max())
        sizes = norm(df[size_col]) * 200

        # Handle colors
        if color_col == 'Spikes':
            colors = df['Spikes'].apply(lambda x: 'blue' if x > 0 else 'grey')
        else:
            color_norm = plt.Normalize(df[color_col].min(), df[color_col].max())
            colors = plt.cm.get_cmap(cmap)(color_norm(df[color_col]))

        # Scatter plot
        scatter = ax.scatter(df[x_col] , df[y_col] ,
                            c=colors, s=sizes,
                            alpha=0.5, edgecolors=None, linewidth=0.5)

        # Add legend
        if color_col == 'Spikes':
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Spike'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='No Spike')
            ]
            ax.legend(handles=legend_elements,
                    loc="upper left",
                    title=f"{legend_title}\n(Circle Size ∝ {size_col})")
        else:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(f'{color_col} (Color)')

            # Add a small text box for size correlation
            text_box = f"Circle Size ∝ {size_col}"
            ax.text(0.02, 0.98, text_box, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Set axis limits and labels
        ax.set_xlim(Plotting._get_lim())
        ax.set_ylim(Plotting._get_lim())
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.show()
        return fig, ax

    @staticmethod
    def plot_scroll_over_video(merged_data: MergedData,
                               columns: list[str],
                               video_path: str,
                               output_path: str):
        Val.validate_type(merged_data, MergedData, "Merged Data")
        Val.validate_type_in_list(columns, str, "Columns")
        Val.validate_path(video_path, file_types=[".mp4", ".avi"])
        Val.validate_path(output_path, file_types=[".mp4", ".avi"])

        df_merged = merged_data.df_merged.copy()

        # Load the video
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set scrolling plot height to 1/5th of video height
        scroll_height = frame_height // 5  
        figsize = (frame_width / 100, scroll_height / 100)

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate,
                              (frame_width, frame_height + scroll_height))

        window_size = 100  # Number of data points visible in scroll window

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create figure for scrolling plot
            fig_scroll, ax_scroll = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)
            if len(columns) == 1:
                ax_scroll = [ax_scroll]

            start_idx = max(0, frame_idx - window_size // 2)
            end_idx = min(len(df_merged), start_idx + window_size)
            data_window = df_merged.iloc[start_idx:end_idx]


            if end_idx == len(df_merged):
                end_xlim = frame_idx + 50
            else:
                end_xlim = end_idx
            if start_idx == 0:
                start_xlim = -50 + frame_idx
                end_xlim = start_xlim + 100
            else:
                start_xlim = start_idx

            for i, col in enumerate(columns):
                ax = ax_scroll[i]
                ax.clear()
                ax.plot(data_window.index, data_window[col], label=col, color=f'C{i}')
                ax.set_ylim(0, df_merged[col].max())
                ax.set_xlim(start_xlim, end_xlim)
                ax.set_ylabel(col, color=f'C{i}')
                ax.axvline(min(start_idx, start_xlim) + window_size // 2, color='black', linestyle='--')
                if col == "Bending_Coefficient":
                    y = merged_data.threshold * df_merged[col].max()
                    ax.axhline(y=y, color='grey', linestyle='--')
                ax.xaxis.set_visible(False)  # Hide x-axis numbers

            # Convert Matplotlib figure to an image
            fig_scroll.canvas.draw()
            scroll_img = np.array(fig_scroll.canvas.renderer.buffer_rgba())[:, :, :3]  # Convert to RGB
            scroll_img = cv2.cvtColor(scroll_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            plt.close(fig_scroll)

            # Resize and combine with video frame
            combined_frame = np.vstack((scroll_img, frame))
            #combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            out.write(combined_frame)

            frame_idx += 1

        cap.release()
        out.release()
