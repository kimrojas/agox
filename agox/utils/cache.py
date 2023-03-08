class CacheNode():
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None

class Cache():
    def __init__(self, max_size=100):
        self.max_size = max_size
        self._table = {}

        self._head = None
        self._tail = None

    def get(self, key, default=None):
        if key not in self._table:
            return default

        node = self._table[key]
        
        if node is not self._tail:
            self._unlink_node(node)
            self._append_node(node)

        return node.value

    def put(self, key, value):
        if key in self._table:
            self._table[key].value = value
            self.get(key)
            return

        if len(self._table) == self.max_size:
            self._table.pop(self._head.key)
            self._remove_head_node()

        new_node = CacheNode(key=key, value=value)
        self._table[key] = new_node
        self._append_node(new_node)
        

    def _unlink_node(self, node):
        if self._head is node:
            self._head = node.next
            if node.next:
                node.next.prev = None
            return

        p, n = node.prev, node.next
        p.next, n.prev = n, p
        

    def _append_node(self, new_node):
        if not self._tail: # first time a node is cached
            self._head = self._tail = new_node
        else:
            self._tail.next = new_node
            new_node.prev = self._tail
            self._tail = self._tail.next
            

    def _remove_head_node(self):
        if not self._head:
            return

        prev = self._head
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        
        del prev
    
    
