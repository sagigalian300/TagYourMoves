def is_point_in_box(point, box):
    """
    Check if a point is inside a bounding box.
    param point: Tuple (x, y) representing the point coordinates.
    param box: List or tuple [x1, y1, x2, y2] representing the bounding box coordinates.
    return: True if the point is inside the box, False otherwise.
    """
    if point is None or box is None:
        return False
        
    px, py = point
    x1, y1, x2, y2 = box
    
    return x1 <= px <= x2 and y1 <= py <= y2