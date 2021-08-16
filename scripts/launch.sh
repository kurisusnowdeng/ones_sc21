#!/bin/sh

usage() {
	echo "Invalid option $1."
	echo "Usage: $0 [-j JOB_NAME] [-n SIZE] [-t TIME]"
	exit -1
}

while getopts 'j:n:t:' OPTION
do
	case ${OPTION} in
		j) JOBNAME=${OPTARG} ;;
		n) SIZE=${OPTARG} ;;
		t) TIME=${OPTARG} ;;
		?) usage ${OPTION} ;;
	esac
done

ROOT_PATH=/path/to/project
LOG_PATH=log/dev

sbatch -J "${JOBNAME}_0" \
	-o "${ROOT_PATH}/log/slurm/${JOBNAME}_0_out.txt" \
	-e "${ROOT_PATH}/log/slurm/${JOBNAME}_0_err.txt" \
	-p rtx -N 1 -n 1 -t ${TIME} \
	${ROOT_PATH}/scripts/master.slurm ${SIZE} \
	"${ROOT_PATH}/${LOG_PATH}/${JOBNAME}_cache"
echo "${JOBNAME}_0 submitted."

if [ ${SIZE} -gt 1 ]
then
	sbatch -J "${JOBNAME}_1" \
		-o "${ROOT_PATH}/log/slurm/${JOBNAME}_1_out.txt" \
		-e "${ROOT_PATH}/log/slurm/${JOBNAME}_1_err.txt" \
		-p rtx -N 1 -n 1 -t ${TIME} \
		${ROOT_PATH}/scripts/worker.slurm \
		"${ROOT_PATH}/${LOG_PATH}/${JOBNAME}_cache"
	echo "${JOBNAME}_1 submitted."

	for ((i=2; ((i*2))<=${SIZE}; i++))
	do
		sbatch -J "${JOBNAME}_${i}" \
			-o "${ROOT_PATH}/log/slurm/${JOBNAME}_${i}_out.txt" \
			-e "${ROOT_PATH}/log/slurm/${JOBNAME}_${i}_err.txt" \
			-p rtx -N 2 -n 2 -t ${TIME} \
			${ROOT_PATH}/scripts/worker.slurm \
			"${ROOT_PATH}/${LOG_PATH}/${JOBNAME}_cache"
		echo "${JOBNAME}_${i} submitted."
	done

fi
