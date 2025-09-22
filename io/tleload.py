# read/parse tle

class TLELoader:
    
    def read_lines(self): # returns list[str]
        lines = []
        for ln in self.path.read_text().splitlines():
            if ln.strip():
                lines.append(ln.rstrip("\n"))

        if len(lines) < 2:
            raise ValueError("TLE must be two lines")
        
        return lines[:2]
