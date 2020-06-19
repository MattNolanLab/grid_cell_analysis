'''
This code is part of MountainSort
https://github.com/flatironinstitute/mountainsort
'''


import numpy as np
import struct

class MdaHeader:
    dt_code=0
    dt='float32'
    num_bytes_per_entry=0
    num_dims=2
    dimprod=1
    dims=[]
    header_size=0
    def __init__(self, dt0, dims0):
        self.dt_code=_dt_code_from_dt(dt0)
        self.dt=dt0
        self.num_bytes_per_entry=_num_bytes_per_entry_from_dt(dt0)
        self.num_dims=len(dims0)
        self.dimprod=np.prod(dims0)
        self.dims=dims0
        self.header_size=(3+len(dims0))*4

class DiskReadMda:
    _path=''
    _header=None
    def __init__(self,path):
        self._path=path
        self._header=_read_header(self._path)
    def dims(self):
        return self._header.dims
    def N1(self):
        return self._header.dims[0]
    def N2(self):
        return self._header.dims[1]
    def N3(self):
        return self._header.dims[2]
    def readChunk(self,i1=-1,i2=-1,i3=-1,N1=1,N2=1,N3=1):
        if (i2<0):
            return self._read_chunk_1d(i1,N1)
        elif (i3<0):
            if N1 != self._header.dims[0]:
                print("Unable to support N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            X=self._read_chunk_1d(i1+N1*i2,N1*N2)
            return np.reshape(X,(N1,N2),order='F')
        else:
            if N1 != self._header.dims[0]:
                print("Unable to support N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            if N2 != self._header.dims[1]:
                print("Unable to support N2 {} != {}".format(N2,self._header.dims[1]))
                return None
            X=self._read_chunk_1d(i1+N1*i2+N1*N2*i3,N1*N2*N3)
            return np.reshape(X,(N1,N2,N3),order='F')
    def _read_chunk_1d(self,i,N):
        f=open(self._path,"rb")
        try:
            f.seek(self._header.header_size+self._header.num_bytes_per_entry*i)
            ret=np.fromfile(f,dtype=self._header.dt,count=N)
            # print(ret.shape)
            f.flush()
            f.close()
            return ret
        except Exception as e: # catch *all* exceptions
            print(e)
            f.flush()
            f.close()
            return None

class DiskWriteMda:
    _path=''
    _header=None
    def __init__(self,path,dims,dt='float64'):
        self._path=path
        self._header=MdaHeader(dt,dims)
        _write_header(path, self._header)
    def N1(self):
        return self._header.dims[0]
    def N2(self):
        return self._header.dims[1]
    def N3(self):
        return self._header.dims[2]
    def writeChunk(self,X,i1=-1,i2=-1,i3=-1):
        if (len(X.shape)>=2):
            N1=X.shape[0]
        else:
            N1=1
        if (len(X.shape)>=2):
            N2=X.shape[1]
        else:
            N2=1
        if (len(X.shape)>=3):
            N3=X.shape[2]
        else:
            N3=1
        if (i2<0):
            return self._write_chunk_1d(X,i1)
        elif (i3<0):
            if N1 != self._header.dims[0]:
                print("Unable to support DiskWriteMda N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            return self._write_chunk_1d(X.ravel(order='F'),i1+N1*i2)
        else:
            if N1 != self._header.dims[0]:
                print("Unable to support DiskWriteMda N1 {} != {}".format(N1,self._header.dims[0]))
                return None
            if N2 != self._header.dims[1]:
                print("Unable to support DiskWriteMda N2 {} != {}".format(N2,self._header.dims[1]))
                return None
            return self._write_chunk_1d(X.ravel(order='F'),i1+N1*i2+N1*N2*i3)
    def _write_chunk_1d(self,X,i):
        N=X.size
        f=open(self._path,"ab")
        try:
            f.seek(self._header.header_size+self._header.num_bytes_per_entry*i)
            X.astype(self._header.dt).tofile(f)
            f.close()
            return True
        except Exception as e: # catch *all* exceptions
            print(e)
            f.close()
            return False

def _dt_from_dt_code(dt_code):
    if dt_code == -2:
        dt='uint8'
    elif dt_code == -3:
        dt='float32'
    elif dt_code == -4:
        dt='int16'
    elif dt_code == -5:
        dt='int32'
    elif dt_code == -6:
        dt='uint16'
    elif dt_code == -7:
        dt='float64'
    elif dt_code == -8:
        dt='uint32'
    else:
        dt=None
    return dt

def _dt_code_from_dt(dt):
    if dt == 'uint8':
        return -2
    if dt == 'float32':
        return -3
    if dt == 'int16':
        return -4
    if dt == 'int32':
        return -5
    if dt == 'uint16':
        return -6
    if dt == 'float64':
        return -7
    if dt == 'uint32':
        return -8
    return None

def _num_bytes_per_entry_from_dt(dt):
    if dt == 'uint8':
        return 1
    if dt == 'float32':
        return 4
    if dt == 'int16':
        return 2
    if dt == 'int32':
        return 4
    if dt == 'uint16':
        return 2
    if dt == 'float64':
        return 8
    if dt == 'uint32':
        return 4
    return None

def _read_header(path):
    f=open(path,"rb")
    try:
        dt_code=_read_int32(f)
        num_bytes_per_entry=_read_int32(f)
        num_dims=_read_int32(f)
        if (num_dims<2) or (num_dims>6):
            print("Invalid number of dimensions: {}".format(num_dims))
            return None
        dims=[]
        dimprod=1
        for j in range(0,num_dims):
            tmp0=_read_int32(f)
            dimprod=dimprod*tmp0
            dims.append(tmp0)
        dt=_dt_from_dt_code(dt_code)
        if dt is None:
            print("Invalid data type code: {}".format(dt_code))
            return None
        # print(dt)
        # print(dims)
        H=MdaHeader(dt,dims)
        f.close()
        return H
    except Exception as e: # catch *all* exceptions
        print(e)
        f.close()
        return None

def _write_header(path,H):
    f=open(path,"wb")
    try:
        _write_int32(f,H.dt_code)
        _write_int32(f,H.num_bytes_per_entry)
        _write_int32(f,H.num_dims)
        for j in range(0,H.num_dims):
            _write_int32(f,H.dims[j])
        f.close()
        return True
    except Exception as e: # catch *all* exceptions
        print(e)
        f.close()
        return False

def readmda(path):
    H=_read_header(path)
    if (H is None):
        print("Problem reading header of: {}".format(path))
        return None
    ret=np.array([])
    f=open(path,"rb")
    try:
        f.seek(H.header_size)
        #This is how I do the column-major order
        ret=np.fromfile(f,dtype=H.dt,count=H.dimprod)
        ret=np.reshape(ret,H.dims,order='F')
        f.flush()
        f.close()
        return ret
    except Exception as e: # catch *all* exceptions
        print(e)
        f.close()
        return None

def writemda32(X,fname):
    num_bytes_per_entry = 4
    return _writemda(X,fname,'float32', num_bytes_per_entry)

def writemda64(X,fname):
    return _writemda(X,fname,'float64')

def writemda8(X,fname):
    return _writemda(X,fname,'uint8')

def writemda32i(X,fname):
    num_bytes_per_entry = 4
    return _writemda(X,fname,'int32', num_bytes_per_entry)

def writemda32ui(X,fname):
    return _writemda(X,fname,'uint32')

def writemda16i(X,fname):
    num_bytes_per_entry = 2
    return _writemda(X,fname,'int16', num_bytes_per_entry)

def writemda16ui(X,fname):
    num_bytes_per_entry = 2
    return _writemda(X,fname,'uint16', num_bytes_per_entry)


def _writeMdaHeader(shapes,f, dt, num_bytes_per_entry):
    """write header of MDA data
    
    Arguments:
        shapes {list} -- list containing size in each dimension
        f {object} -- file handle
        dt {np.dtype} -- numpy datatype
        num_bytes_per_entry {int} -- number of bype per data
    
    Returns:
        object -- file object
    """

    dt_code=0
    #num_bytes_per_entry=2 # changed 0 to 2 here
    dt_code=_dt_code_from_dt(dt)
    if dt_code is None:
        print("Unexpected data type: {}".format(dt))
        return False

    try:
        _write_int32(f, dt_code)
        _write_int32(f, num_bytes_per_entry)
        _write_int32(f, len(shapes))
        for s in shapes:
            _write_int32(f, s)
    except Exception as e: # catch *all* exceptions
        print(e)
    finally:
        f.flush()
        # f.close()
        return f

def _writeMdaData(f,X):
    """Write raw data into MDA file, allow repeated write
    
    Arguments:
        f {file handle} -- file handle of the mda file
        X {np.narray} -- 1d array to be written, it will be intepreted as a column of matrix
    """
    #This is how I do column-major order
    A = np.reshape(X, X.size, order='F').astype(X.dtype)
    A.tofile(f)
    f.flush()

def writeMdaByChunk(x,fname,chunk=300000): 
    """Write MDA file to disk by chunk
    
    Arguments:
        x {np.narray} -- data to be written, should be in channel x time format
        fname {str} -- path of the MDA file
        chunk {int} -- the size of each chunk to be written
    
    Raises:
        ValueError: error if the file is not in channel x time format
    """
    if x.shape[0]>x.shape[1]:
        raise ValueError('The number of channel is larger than the length of data. The signal probably is not in a correct format for mountainsort')

    f = open(fname,'wb')
    f=_writeMdaHeader([x.shape[0],x.shape[1]],f,x.dtype,x.dtype.itemsize)

    for i in range((x.shape[1]//chunk)+1): #write data in chunk, and make sure the last chunk is coverred
        _writeMdaData(f,x[:,i*chunk:(i+1)*chunk])

    f.close()

def _writemda(X,fname,dt, num_bytes_per_entry):
    dt_code=0
    #num_bytes_per_entry=2 # changed 0 to 2 here
    dt_code=_dt_code_from_dt(dt)
    if dt_code is None:
        print("Unexpected data type: {}".format(dt))
        return False

    f=open(fname,'wb')
    try:
        _write_int32(f, dt_code)
        _write_int32(f, num_bytes_per_entry)
        _write_int32(f, X.ndim)
        for j in range(0, X.ndim):
            _write_int32(f, X.shape[j])
        #This is how I do column-major order
        A = np.reshape(X, X.size, order='F').astype(dt)
        A.tofile(f)
    except Exception as e: # catch *all* exceptions
        print(e)
    finally:
        f.flush()
        f.close()
        return True

def _read_int32(f):
    return struct.unpack('<i',f.read(4))[0]

def _write_int32(f,val):
    f.write(struct.pack('<i',val))

def mdaio_test():
    M=4
    N=12
    X=np.ndarray((M,N))
    for n in range(0,N):
        for m in range(0,M):
            X[m,n]=n*10+m
    writemda32(X,'tmp1.mda')
    Y=readmda('tmp1.mda')
    # print(Y)
    # print(np.absolute(X-Y).max())
    Z=DiskReadMda('tmp1.mda')
    # print(Z.readChunk(i1=0,i2=4,N1=M,N2=N-4))

    A=DiskWriteMda('tmpA.mda',(M,N))
    A.writeChunk(Y,i1=0,i2=0)
    B=readmda('tmpA.mda')
    # print(B.shape)
    # print(B)

