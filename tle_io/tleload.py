# read/parse tle

class TLELoader:

    @staticmethod
    def read_lines(text): # returns list[str]
        lines = []
        for ln in text.splitlines():
            if ln.strip():
                lines.append(ln.rstrip("\n"))

        if len(lines) < 2:
            raise ValueError("TLE must be two lines")
        
        return lines[:2]
