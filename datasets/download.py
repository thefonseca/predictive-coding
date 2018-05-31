from urllib import urlretrieve
from tqdm import tqdm
import os, errno
import zipfile
import argparse

import download

FLAGS = None

def check_makedir(path):
    '''
    Creates a directory if it does not exist.
    '''
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def extract_zip(filename, dest_dir=None):
    '''
    Extracts a zip file to specified folder.
    '''

    print('Extracting {}...'.format(filename))

    if not dest_dir:
        dest_dir = os.path.splitext(filename)[0]
        #dest_dir = os.path.join('.', dest_dir)
        
    if os.path.exists(dest_dir):
        print('File already extracted to {}'.format(dest_dir))
        return
        
    with zipfile.ZipFile(filename) as f:
        f.extractall(dest_dir)
        extracted_file = os.path.join(dest_dir, f.namelist()[0])
        print('File extracted to {}'.format(extracted_file))
        return extracted_file


def urlopen_with_progress(url, dest_filename):
    def my_hook(t):
        """
        Wraps tqdm instance. Don't forget to close() or __exit__() the tqdm instance
        once you're done (easiest using a context manager, eg: `with` syntax)
        Example
        -------
        >>> with tqdm(...) as t:
        ...     reporthook = my_hook(t)
        ...     urllib.urlretrieve(..., reporthook=reporthook)
        """
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            """
            b     : int, optional    Number of blocks just transferred [default: 1]
            bsize : int, optional    Size of each block (in tqdm units) [default: 1]
            tsize : int, optional    Total size (in tqdm units). If [default: None]
                                     remains unchanged.
            """
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner

    with tqdm(unit='B', unit_scale=True, miniters=1,
              desc="Downloading file...") as t:
        return urlretrieve(url, dest_filename, reporthook=my_hook(t))

    #with open(filename, 'r') as f:
    #    return f.read()


def maybe_download(url, filename=None, expected_bytes=None, force=False, data_root='.'):
    """Download a file if not present, and make sure it's the right size."""
    
    if filename is None:
        filename = url.rsplit('/', 1)[-1]
    
    dest_filename = os.path.join(data_root, filename)
    check_makedir(dest_filename)

    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        #filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        filename, _ = urlopen_with_progress(url.format(filename), dest_filename)
        print('\nDownload Complete!')

        statinfo = os.stat(dest_filename)
        
        if expected_bytes is None:
            print('Found {} (size not verified)'.format(dest_filename))
        elif expected_bytes is None or statinfo.st_size == expected_bytes:
            print('Found and verified', dest_filename)
        else:
            raise Exception(
                'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')

    elif os.path.exists(dest_filename):
        print('File already exists: {}'.format(dest_filename))
        
    return dest_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download file from URL.')
    parser.add_argument('url', help='URL to download from')
    parser.add_argument('--dest_filename', help='Filename for downloaded file')
    parser.add_argument('--dest_dir', default='.', help='Directory for storing downloaded file')
    parser.add_argument('--expected_bytes', type=int, help='Expected size for downloaded file')
    parser.add_argument("--unzip", help="Unzip the downloaded file", 
                        action="store_true")
    FLAGS, unparsed = parser.parse_known_args()
    
    downloaded_file = maybe_download(FLAGS.url, FLAGS.dest_filename, 
                                expected_bytes=FLAGS.expected_bytes, 
                                data_root=FLAGS.dest_dir)
    
    if FLAGS.unzip:
        download.extract_zip(downloaded_file)