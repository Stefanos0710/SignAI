from .updater import Updater

if __name__ == '__main__':
    u = Updater()
    print('API_URL:', u.API_URL)
    print('HEADERS present:', 'Authorization' in u.HEADERS and bool(u.HEADERS['Authorization']))
    print('BACKUP_DIR:', u.BACKUP_DIR)
    print('UPDATE_DIR:', u.UPDATE_DIR)
    print('UPDATE_FILE:', u.UPDATE_FILE)
    print('save_files:', u.save_files)
