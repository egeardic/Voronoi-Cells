# Interactive Bounded Voronoi Diagram

This project is a Python application that lets you **create, move, and delete points** in real-time, generating a **bounded Voronoi diagram** where each cell is uniquely colored.  

It’s built with **pygame**, **shapely**, and **scipy**.

---

## Features
- **Add points**: Left-click anywhere to create a new point (random color assigned).  
- **Move points**: Left-click & drag an existing point to move it.  
- **Delete points**: Right-click on a point to remove it.  
- **Live updates**: Voronoi diagram updates immediately as points move.  
- **Finite borders**: All Voronoi cells are clipped to the window rectangle.  
- **Randomized colors**: Each point generates a distinct Voronoi region with its own color.  

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/egeardic/Voronoi-Cells.git
cd Voronoi-Cells
pip install -r requirements.txt