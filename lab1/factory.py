class BlockFac:
    def __init__(self, name):
        self.name = name
        self.ChildBlocks = []

        self.ChildBlocks1 = BlockFac('ChildBlock1')
        self.ChildBlocks2 = BlockFac('ChildBlock2')
        self.ChildBlocks.append(self.ChildBlocks1)
        self.ChildBlocks.append(self.ChildBlocks2)

        self.Exps = []
        self.Exps1 = ExpFac()
        self.Exps2 = ExpFac()
        self.Exps.append(self.Exps1)
        self.Exps.append(self.Exps2)


    def buildBlock(self,blocks,exps):
        block = Block('Block')

        return Block('Block')
    
class ExpFac:
    def __init__(self):
        self.lval = "1"
        self.rval = "1"
        self.op = '=='

    def buildExp():
        pass
 
    def tostring(self):
        return self.lval + self.op + self.rval


class forLoopFac:
    def __init__(self):
        self.init = 0
        self.condition = 0
        self.update = 0
        self.block = BlockFac('forLoop')

    def tostring(self):
        return 'for(' + self.init + ';' + self.condition + ';' + self.update + ')' + self.block.tostring()


class Block:
    def __init__(self):
        self.ChildBlocks = []
        self.Exps = []

    def tostring(self):
        return '{\n' + self.ChildBlocks + '\n' + self.Exps + '\n}'

int main(){
    for (int i = 0; i < 10; i++){
        1 == 1
    }
    {
        
    }

}

class Parser:
    def __init__(self):
        self.tokens = []

        # abstract syntax tree
        self.AST = []
        # vis AST

        # Block
        self.BlockStack = []


class 