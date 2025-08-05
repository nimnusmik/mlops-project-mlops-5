import subprocess
from dotenv import load_dotenv

# 현재 디렉터리에서 .env 파일을 찾아 환경 변수를 로드합니다.
# 이 함수가 실행되면 .env 파일의 내용이 os.environ에 추가됩니다.
load_dotenv()

# feast 명령어를 실행할 때 필요한 환경 변수들이 이미 설정된 상태입니다.
# feast apply 명령어 실행
print("Feast apply 명령어를 실행합니다...")
result = subprocess.run(["feast", "apply", "--repo", "../.."], capture_output=True, text=True)

# 실행 결과 출력
print("--- stdout ---")
print(result.stdout)
print("--- stderr ---")
print(result.stderr)
print("----------------")